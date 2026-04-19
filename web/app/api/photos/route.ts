import { NextResponse } from "next/server";
import { getPhotoCollection } from "@/lib/photo-collection";

const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const nominatimContact =
  process.env.NOMINATIM_CONTACT_EMAIL ?? "hackku-2026@example.com";

async function geocodeAddress(address: string) {
  const geocodeUrl = new URL("https://nominatim.openstreetmap.org/search");
  geocodeUrl.searchParams.set("q", address);
  geocodeUrl.searchParams.set("format", "jsonv2");
  geocodeUrl.searchParams.set("limit", "1");
  geocodeUrl.searchParams.set("addressdetails", "1");

  const response = await fetch(geocodeUrl, {
    cache: "no-store",
    headers: {
      "User-Agent": `HackKU-2026/0.1 (${nominatimContact})`,
      "Accept-Language": "en-US,en;q=0.9",
    },
  });

  if (!response.ok) {
    throw new Error("OpenStreetMap geocoding request failed.");
  }

  const payload = (await response.json()) as Array<{
    display_name?: string;
    lat?: string;
    lon?: string;
  }>;

  const result = payload[0];
  const latitude = result?.lat ? Number(result.lat) : Number.NaN;
  const longitude = result?.lon ? Number(result.lon) : Number.NaN;

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    throw new Error("Address could not be geocoded.");
  }

  return {
    formattedAddress:
      typeof result?.display_name === "string"
        ? result.display_name
        : address,
    latitude,
    longitude,
  };
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const photo = formData.get("photo");
    const addressValue = formData.get("address");
    const address =
      typeof addressValue === "string" ? addressValue.trim() : "";

    if (!(photo instanceof File)) {
      return NextResponse.json({ error: "No file received." }, { status: 400 });
    }

    if (!ALLOWED_TYPES.has(photo.type)) {
      return NextResponse.json(
        { error: "Use a JPG, PNG, or WEBP image." },
        { status: 400 },
      );
    }

    if (photo.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "File is too large. Keep it under 10 MB." },
        { status: 400 },
      );
    }

    if (!address) {
      return NextResponse.json(
        { error: "Enter an address for this pothole report." },
        { status: 400 },
      );
    }

    const geocodedAddress = await geocodeAddress(address);
    const buffer = Buffer.from(await photo.arrayBuffer());
    const collection = await getPhotoCollection();

    const result = await collection.insertOne({
      filename: photo.name,
      contentType: photo.type,
      size: photo.size,
      uploadedAt: new Date(),
      image: buffer,
      analysisStatus: "pending",
      address: geocodedAddress.formattedAddress,
      location: {
        latitude: geocodedAddress.latitude,
        longitude: geocodedAddress.longitude,
      },
    });

    return NextResponse.json({
      id: result.insertedId.toString(),
      filename: photo.name,
    });
  } catch (error) {
    console.error("Unable to save photo to MongoDB.", error);

    const message =
      error instanceof Error ? error.message : "Unexpected upload error.";

    return NextResponse.json(
      {
        error:
          message === "Address could not be geocoded."
            ? "That address could not be matched. Try a more specific address."
            : "The report could not be saved. Verify MongoDB and OpenStreetMap geocoding access.",
      },
      { status: 503 },
    );
  }
}
