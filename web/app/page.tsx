import { ObjectId } from "mongodb";
import nextDynamic from "next/dynamic";
import Image from "next/image";
import { UploadForm } from "@/components/upload-form";
import { getPhotoCollection } from "@/lib/photo-collection";

export const dynamic = "force-dynamic";

const PotholeMap = nextDynamic(
  () => import("@/components/pothole-map").then((module) => module.PotholeMap),
  {
    ssr: false,
  },
);

type PotholeItem = {
  _id: string;
  filename: string;
  contentType: string;
  uploadedAt: string;
  size: number;
  address?: string;
  latitude?: number;
  longitude?: number;
};

async function getPhotos() {
  try {
    const collection = await getPhotoCollection();

    const photos = await collection
      .find(
        {},
        {
          projection: {
            filename: 1,
            contentType: 1,
            uploadedAt: 1,
            size: 1,
            address: 1,
            location: 1,
          },
        },
      )
      .sort({ uploadedAt: -1 })
      .limit(12)
      .toArray();

    return {
      photos: photos.map((photo) => {
        const latitude =
          typeof photo.location?.latitude === "number"
            ? photo.location.latitude
            : undefined;
        const longitude =
          typeof photo.location?.longitude === "number"
            ? photo.location.longitude
            : undefined;

        return {
          _id: (photo._id as ObjectId).toString(),
          filename: String(photo.filename),
          contentType: String(photo.contentType),
          uploadedAt: new Date(photo.uploadedAt).toLocaleString(),
          size: Number(photo.size),
          address:
            typeof photo.address === "string" ? photo.address : undefined,
          latitude,
          longitude,
        };
      }),
      loadError: "",
    };
  } catch (error) {
    console.error("Unable to load photos from MongoDB.", error);

    return {
      photos: [] as PotholeItem[],
      loadError:
        "The app could not reach MongoDB right now. Check the connection string, Atlas network access, and TLS settings.",
    };
  }
}

function formatFileSize(size: number) {
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }

  return `${Math.max(1, Math.round(size / 1024))} KB`;
}

function formatAddress(photo: PotholeItem) {
  return photo.address ?? "Address not added yet";
}

export default async function HomePage() {
  const { photos, loadError } = await getPhotos();
  const mappedPhotos = photos.filter(
    (photo) => photo.latitude !== undefined && photo.longitude !== undefined,
  );

  return (
    <main
      style={{
        padding: "40px 16px 80px",
      }}
    >
      <section
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          display: "grid",
          gap: 24,
        }}
      >
        <section
          style={{
            background: "var(--card)",
            border: "1px solid var(--card-border)",
            boxShadow: "var(--shadow)",
            borderRadius: 30,
            padding: "28px",
            backdropFilter: "blur(8px)",
            display: "grid",
            gap: 20,
          }}
        >
          <p
            style={{
              margin: 0,
              fontSize: 13,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "var(--accent-dark)",
            }}
          >
            Street Wear Monitor
          </p>
          <h1
            style={{
              margin: 0,
              fontSize: "clamp(2.6rem, 7vw, 5.8rem)",
              lineHeight: 0.92,
              maxWidth: 860,
            }}
          >
            Log potholes, pin damage, and map the trouble spots.
          </h1>
          <p
            style={{
              margin: 0,
              maxWidth: 760,
              fontSize: 18,
              lineHeight: 1.7,
              color: "var(--muted)",
            }}
          >
            Capture road damage with a photo, attach a street address, and keep
            the field team aligned with one upload queue, one pothole list, and
            one live map.
          </p>
        </section>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 24,
          }}
        >
          <div
            style={{
              background: "var(--card)",
              border: "1px solid var(--card-border)",
              boxShadow: "var(--shadow)",
              borderRadius: 28,
              padding: "28px",
            }}
          >
            <h2 style={{ marginTop: 0, marginBottom: 8 }}>Submit a Report</h2>
            <p
              style={{
                marginTop: 0,
                marginBottom: 20,
                color: "var(--muted)",
                lineHeight: 1.6,
              }}
            >
              Add a road-damage image, enter the street address, and push it into
              the pothole queue.
            </p>
            <UploadForm />

            {loadError ? (
              <p
                style={{
                  marginTop: 16,
                  marginBottom: 0,
                  color: "#365f9d",
                  lineHeight: 1.6,
                }}
              >
                {loadError}
              </p>
            ) : null}
          </div>

          <div
            style={{
              background: "var(--card)",
              border: "1px solid var(--card-border)",
              boxShadow: "var(--shadow)",
              borderRadius: 28,
              padding: "28px",
              display: "grid",
              gap: 18,
              alignContent: "start",
              maxHeight: 620,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                flexWrap: "wrap",
                gap: 12,
              }}
            >
              <div style={{ display: "grid", gap: 6 }}>
                <h2 style={{ margin: 0 }}>Pothole List</h2>
                <span style={{ color: "var(--muted)", fontSize: 14 }}>
                  {photos.length} tracked
                </span>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                  gap: 14,
                  flex: "1 1 320px",
                }}
              >
                <div
                  style={{
                    borderRadius: 22,
                    padding: "18px 20px",
                    background: "rgba(255,255,255,0.62)",
                    border: "1px solid var(--card-border)",
                  }}
                >
                  <div style={{ fontSize: 13, color: "var(--muted)" }}>
                    Logged Cases
                  </div>
                  <div style={{ fontSize: 32, fontWeight: 700 }}>{photos.length}</div>
                </div>
                <div
                  style={{
                    borderRadius: 22,
                    padding: "18px 20px",
                    background: "rgba(255,255,255,0.62)",
                    border: "1px solid var(--card-border)",
                  }}
                >
                  <div style={{ fontSize: 13, color: "var(--muted)" }}>
                    Map Pins
                  </div>
                  <div style={{ fontSize: 32, fontWeight: 700 }}>
                    {mappedPhotos.length}
                  </div>
                </div>
              </div>
            </div>

            {photos.length === 0 ? (
              <div
                style={{
                  borderRadius: 20,
                  border: "1px dashed var(--card-border)",
                  padding: "28px",
                  color: "var(--muted)",
                  lineHeight: 1.7,
                }}
              >
                {loadError
                  ? "Pothole data is temporarily unavailable because MongoDB could not be reached."
                  : "No potholes logged yet. Submit one on the left to start the field list."}
              </div>
            ) : (
              <div
                style={{
                  display: "grid",
                  gap: 14,
                  overflowY: "auto",
                  paddingRight: 6,
                }}
              >
                {photos.map((photo) => (
                  <article
                    key={photo._id}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "88px minmax(0, 1fr)",
                      gap: 14,
                      padding: "14px",
                      borderRadius: 22,
                      border: "1px solid var(--card-border)",
                      background: "rgba(255,255,255,0.54)",
                    }}
                  >
                    <div
                      style={{
                        position: "relative",
                        aspectRatio: "1 / 1",
                        borderRadius: 18,
                        overflow: "hidden",
                        background: "#d8e8fb",
                      }}
                    >
                      <Image
                        src={`/api/photos/${photo._id}`}
                        alt={photo.filename}
                        fill
                        sizes="88px"
                        style={{ objectFit: "cover" }}
                      />
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 16,
                          fontWeight: 700,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                        title={photo.filename}
                      >
                        {photo.filename}
                      </div>
                      <div
                        style={{
                          marginTop: 4,
                          color: "var(--muted)",
                          fontSize: 14,
                          lineHeight: 1.5,
                        }}
                      >
                        {formatFileSize(photo.size)} · {photo.uploadedAt}
                      </div>
                      <div
                        style={{
                          marginTop: 6,
                          fontSize: 14,
                          color: "var(--text)",
                          lineHeight: 1.5,
                        }}
                      >
                        {formatAddress(photo)}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>

        <section
          style={{
            background: "var(--card)",
            border: "1px solid var(--card-border)",
            boxShadow: "var(--shadow)",
            borderRadius: 30,
            padding: "28px",
            display: "grid",
            gap: 18,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            <div>
              <h2 style={{ margin: 0 }}>Street Damage Map</h2>
              <p
                style={{
                  margin: "6px 0 0",
                  color: "var(--muted)",
                  lineHeight: 1.6,
                }}
              >
                Pins appear here for every report with a matched street address.
              </p>
            </div>
            <div style={{ color: "var(--muted)", fontSize: 14 }}>
              {mappedPhotos.length} mapped reports
            </div>
          </div>

          <PotholeMap
            items={mappedPhotos.map((photo) => ({
              id: photo._id,
              title: photo.filename,
              uploadedAt: photo.uploadedAt,
              address: photo.address,
              latitude: photo.latitude ?? 0,
              longitude: photo.longitude ?? 0,
            }))}
          />
        </section>
      </section>
    </main>
  );
}
