"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import {
  formatClassifierLabel,
  formatSeverityLabel,
} from "@/lib/classification-labels";

type BrowserLocation = {
  latitude: number;
  longitude: number;
  accuracy: number;
};

type UploadReport = {
  id: string;
  filename: string;
  address?: string;
  analysisSummary?: string;
  authorityToReport?: string;
  analysisStatus?: string;
  analysisNote?: string;
  classifier?: {
    prediction: string;
    confidence: number;
    severity: string;
    recommendedAction: string;
  };
  error?: string;
};

function formatConfidence(confidence: number) {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function UploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [location, setLocation] = useState<BrowserLocation | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [latestReport, setLatestReport] = useState<UploadReport | null>(null);
  const [status, setStatus] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setStatus("");
    setLatestReport(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : "");
  }

  function requestLocation() {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      throw new Error("Browser geolocation is unavailable on this device.");
    }

    return new Promise<BrowserLocation>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const nextLocation = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy,
          };
          setLocation(nextLocation);
          resolve(nextLocation);
        },
        () => {
          reject(
            new Error(
              "Location permission is required so RoadScout can geocode the report.",
            ),
          );
        },
        {
          enableHighAccuracy: true,
          timeout: 12_000,
        },
      );
    });
  }

  async function handleGetLocation() {
    setIsLocating(true);
    setStatus("Requesting your location...");

    try {
      await requestLocation();
      setStatus("Location captured. You can upload now.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to get your location.";
      setStatus(message);
    } finally {
      setIsLocating(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;

    if (!file) {
      setStatus("Choose a photo first.");
      return;
    }

    setIsSubmitting(true);
    setStatus("Requesting your location...");

    try {
      const activeLocation = location ?? (await requestLocation());
      const formData = new FormData();
      formData.append("photo", file);
      formData.append("latitude", String(activeLocation.latitude));
      formData.append("longitude", String(activeLocation.longitude));
      formData.append("accuracy", String(activeLocation.accuracy));

      setStatus("Uploading and running AI analysis...");

      const response = await fetch("/api/photos", {
        method: "POST",
        body: formData,
      });

      const payload = (await response.json()) as UploadReport;

      if (!response.ok) {
        throw new Error(payload.error ?? "Upload failed.");
      }

      setLatestReport(payload);

      setStatus(
        payload.analysisStatus === "complete"
          ? "Upload complete. AI analysis is ready."
          : "Upload complete. Classifier result is ready.",
      );
      setFile(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
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
          border: "1px dashed rgba(255, 205, 103, 0.45)",
          borderRadius: 22,
          padding: 18,
          background: "rgba(33, 33, 33, 0.82)",
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
              border: "1px solid rgba(255, 205, 103, 0.26)",
            }}
          />
        </div>
      ) : null}

      <section
        style={{
          border: "1px solid var(--card-border)",
          borderRadius: 20,
          padding: 14,
          background: "rgba(20, 20, 20, 0.74)",
          display: "grid",
          gap: 10,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <strong>Location for geocoding</strong>
          <button
            type="button"
            onClick={handleGetLocation}
            disabled={isLocating || isSubmitting}
            style={{
              appearance: "none",
              border: "1px solid rgba(255, 205, 103, 0.45)",
              background: "rgba(255, 205, 103, 0.15)",
              color: "var(--text)",
              borderRadius: 999,
              padding: "8px 14px",
              fontWeight: 700,
              cursor: isLocating || isSubmitting ? "not-allowed" : "pointer",
            }}
          >
            {isLocating
              ? "Locating..."
              : location
                ? "Refresh Location"
                : "Use Current Location"}
          </button>
        </div>

        <p
          style={{
            margin: 0,
            color: "var(--muted)",
            lineHeight: 1.5,
            fontSize: 14,
          }}
        >
          {location
            ? `Captured: ${location.latitude.toFixed(5)}, ${location.longitude.toFixed(5)} (+/-${Math.round(location.accuracy)}m)`
            : "RoadScout will prompt your browser for location permission during upload."}
        </p>
      </section>

      <button
        type="submit"
        disabled={isSubmitting || isLocating}
        style={{
          appearance: "none",
          border: 0,
          borderRadius: 999,
          padding: "14px 18px",
          background: isSubmitting ? "#6f5e39" : "var(--accent)",
          color: "#161616",
          fontWeight: 700,
          cursor: isSubmitting ? "progress" : "pointer",
        }}
      >
        {isSubmitting ? "Processing..." : "Upload and Analyze"}
      </button>

      <p
        aria-live="polite"
        style={{
          margin: 0,
          minHeight: 24,
          fontSize: 14,
          lineHeight: 1.4,
          color:
            status.includes("complete") || status.includes("captured")
              ? "var(--accent-dark)"
              : "var(--muted)",
        }}
      >
        {status}
      </p>

      {latestReport ? (
        <section
          style={{
            border: "1px solid var(--card-border)",
            borderRadius: 20,
            padding: 16,
            background: "rgba(19, 19, 19, 0.86)",
            display: "grid",
            gap: 8,
          }}
        >
          <h3 style={{ margin: 0, fontSize: 17 }}>Latest AI Output</h3>

          {latestReport.classifier ? (
            <>
              <p style={{ margin: 0, lineHeight: 1.5 }}>
                <strong>Class:</strong>{" "}
                {formatClassifierLabel(latestReport.classifier.prediction)}
              </p>
              <p style={{ margin: 0, lineHeight: 1.5 }}>
                <strong>Confidence:</strong>{" "}
                {formatConfidence(latestReport.classifier.confidence)}
              </p>
              <p style={{ margin: 0, lineHeight: 1.5 }}>
                <strong>Severity:</strong>{" "}
                {formatSeverityLabel(latestReport.classifier.severity)}
              </p>
              <p style={{ margin: 0, lineHeight: 1.5 }}>
                <strong>Action:</strong>{" "}
                {latestReport.classifier.recommendedAction}
              </p>
            </>
          ) : null}

          {latestReport.analysisSummary ? (
            <p style={{ margin: 0, lineHeight: 1.6 }}>
              <strong>Gemini Brief:</strong> {latestReport.analysisSummary}
            </p>
          ) : null}

          {latestReport.authorityToReport ? (
            <p style={{ margin: 0, lineHeight: 1.6 }}>
              <strong>Report To:</strong> {latestReport.authorityToReport}
            </p>
          ) : null}

          {latestReport.address ? (
            <p style={{ margin: 0, lineHeight: 1.6, color: "var(--muted)" }}>
              <strong>Address:</strong> {latestReport.address}
            </p>
          ) : null}
        </section>
      ) : null}
    </form>
  );
}
