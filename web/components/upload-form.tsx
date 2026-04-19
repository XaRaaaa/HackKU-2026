"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { useRouter } from "next/navigation";

export function UploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [address, setAddress] = useState("");
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

    if (!address.trim()) {
      setStatus("Enter an address for the report.");
      return;
    }

    setIsSubmitting(true);
    setStatus("Uploading...");

    try {
      const formData = new FormData();
      formData.append("photo", file);
      formData.append("address", address);

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
      setAddress("");
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
          border: "1px dashed rgba(35, 103, 200, 0.35)",
          borderRadius: 22,
          padding: 18,
          background: "rgba(255,255,255,0.5)",
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
              border: "1px solid rgba(35, 103, 200, 0.12)",
            }}
          />
        </div>
      ) : null}

      <label style={{ display: "grid", gap: 6 }}>
        <span style={{ display: "block", height: 4 }} aria-hidden="true" />
        <span style={{ fontWeight: 700 }}>Street Address</span>
        <input
          name="address"
          type="text"
          autoComplete="street-address"
          placeholder="1450 Jayhawk Blvd, Lawrence, KS"
          value={address}
          onChange={(event) => setAddress(event.target.value)}
          style={{
            border: "1px solid rgba(35, 103, 200, 0.22)",
            borderRadius: 16,
            padding: "12px 14px",
            background: "rgba(255,255,255,0.72)",
          }}
        />
      </label>

      <button
        type="submit"
        disabled={isSubmitting}
        style={{
          appearance: "none",
          border: 0,
          borderRadius: 999,
          padding: "14px 18px",
          background: isSubmitting ? "#7fa6de" : "var(--accent)",
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
          minHeight: 18,
          fontSize: 14,
          lineHeight: 1.4,
          color: status === "Upload complete." ? "var(--accent-dark)" : "var(--muted)",
        }}
      >
        {status}
      </p>
    </form>
  );
}
