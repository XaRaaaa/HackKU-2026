import { ObjectId } from "mongodb";
import Image from "next/image";
import { getPhotoCollection } from "@/lib/photo-collection";
import { UploadForm } from "@/components/upload-form";

export const dynamic = "force-dynamic";

async function getPhotos() {
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
        },
      },
    )
    .sort({ uploadedAt: -1 })
    .limit(12)
    .toArray();

  return photos.map((photo) => ({
    _id: (photo._id as ObjectId).toString(),
    filename: String(photo.filename),
    contentType: String(photo.contentType),
    uploadedAt: new Date(photo.uploadedAt).toLocaleString(),
    size: Number(photo.size),
  }));
}

function formatFileSize(size: number) {
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }

  return `${Math.max(1, Math.round(size / 1024))} KB`;
}

export default async function HomePage() {
  const photos = await getPhotos();

  return (
    <main
      style={{
        padding: "48px 20px 80px",
      }}
    >
      <section
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          display: "grid",
          gap: 28,
        }}
      >
        <div
          style={{
            background: "var(--card)",
            border: "1px solid var(--card-border)",
            boxShadow: "var(--shadow)",
            borderRadius: 28,
            padding: "32px",
            backdropFilter: "blur(8px)",
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
            Phase 1
          </p>
          <h1
            style={{
              margin: "12px 0 10px",
              fontSize: "clamp(2.4rem, 6vw, 4.8rem)",
              lineHeight: 0.95,
              maxWidth: 700,
            }}
          >
            Asphalt photo intake for future analysis.
          </h1>
          <p
            style={{
              margin: 0,
              maxWidth: 680,
              fontSize: 18,
              lineHeight: 1.6,
              color: "var(--muted)",
            }}
          >
            Upload images now, keep them in MongoDB, and leave a clean place to
            plug in model inference later.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
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
            <h2 style={{ marginTop: 0, marginBottom: 8 }}>Upload a Photo</h2>
            <p
              style={{
                marginTop: 0,
                marginBottom: 20,
                color: "var(--muted)",
                lineHeight: 1.6,
              }}
            >
              This starter accepts `jpg`, `jpeg`, `png`, and `webp` files and
              stores them directly in MongoDB.
            </p>
            <UploadForm />
          </div>

          <div
            style={{
              background: "var(--card)",
              border: "1px solid var(--card-border)",
              boxShadow: "var(--shadow)",
              borderRadius: 28,
              padding: "28px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                gap: 12,
                marginBottom: 20,
              }}
            >
              <h2 style={{ margin: 0 }}>Recent Uploads</h2>
              <span style={{ color: "var(--muted)", fontSize: 14 }}>
                {photos.length} saved
              </span>
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
                No photos yet. Upload one on the left to confirm the pipeline is
                working.
              </div>
            ) : (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: 18,
                }}
              >
                {photos.map((photo) => (
                  <article
                    key={photo._id}
                    style={{
                      display: "grid",
                      gap: 10,
                    }}
                  >
                    <div
                      style={{
                        position: "relative",
                        aspectRatio: "4 / 3",
                        borderRadius: 18,
                        overflow: "hidden",
                        background: "#eadcc8",
                      }}
                    >
                      <Image
                        src={`/api/photos/${photo._id}`}
                        alt={photo.filename}
                        fill
                        sizes="(max-width: 900px) 100vw, 33vw"
                        style={{ objectFit: "cover" }}
                      />
                    </div>
                    <div>
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
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
