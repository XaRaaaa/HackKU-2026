"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { useRouter } from "next/navigation";

export function UploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setStatus("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : "");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;

    if (!file) {
      setStatus("Choose a photo first.");
      return;
    }

    setIsSubmitting(true);
    setStatus("Uploading...");

    try {
      const formData = new FormData();
      formData.append("photo", file);

      const response = await fetch("/api/photos", {
        method: "POST",
        body: formData,
      });

      const payload = (await response.json()) as { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "Upload failed.");
      }

      setStatus("Upload complete.");
      setFile(null);
      setPreviewUrl("");
      form.reset();
      router.refresh();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected upload error.";
      setStatus(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: 16 }}>
      <label
        htmlFor="photo"
        style={{
          border: "1px dashed rgba(96, 78, 49, 0.35)",
          borderRadius: 22,
          padding: 18,
          background: "rgba(255,255,255,0.45)",
          display: "grid",
          gap: 12,
        }}
      >
        <span style={{ fontWeight: 700 }}>Select a photo</span>
        <input
          id="photo"
          name="photo"
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          onChange={handleFileChange}
        />
      </label>

      {previewUrl ? (
        <div style={{ display: "grid", gap: 10 }}>
          <span style={{ fontWeight: 700 }}>Preview</span>
          <img
            src={previewUrl}
            alt="Selected preview"
            style={{
              width: "100%",
              maxHeight: 260,
              objectFit: "cover",
              borderRadius: 22,
              border: "1px solid rgba(96, 78, 49, 0.12)",
            }}
          />
        </div>
      ) : null}

      <button
        type="submit"
        disabled={isSubmitting}
        style={{
          appearance: "none",
          border: 0,
          borderRadius: 999,
          padding: "14px 18px",
          background: isSubmitting ? "#caa08f" : "var(--accent)",
          color: "white",
          fontWeight: 700,
          cursor: isSubmitting ? "progress" : "pointer",
        }}
      >
        {isSubmitting ? "Uploading..." : "Save Photo"}
      </button>

      <p
        aria-live="polite"
        style={{
          margin: 0,
          minHeight: 24,
          color: status === "Upload complete." ? "var(--accent-dark)" : "var(--muted)",
        }}
      >
        {status}
      </p>
    </form>
  );
}
