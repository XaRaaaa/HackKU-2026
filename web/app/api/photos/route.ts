import { NextResponse } from "next/server";
import { getPhotoCollection } from "@/lib/photo-collection";

const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const MAX_FILE_SIZE = 10 * 1024 * 1024;

export async function POST(request: Request) {
  const formData = await request.formData();
  const photo = formData.get("photo");

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

  const buffer = Buffer.from(await photo.arrayBuffer());
  const collection = await getPhotoCollection();

  const result = await collection.insertOne({
    filename: photo.name,
    contentType: photo.type,
    size: photo.size,
    uploadedAt: new Date(),
    image: buffer,
    analysisStatus: "pending",
  });

  return NextResponse.json({
    id: result.insertedId.toString(),
    filename: photo.name,
  });
}
