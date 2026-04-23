import { NextResponse } from "next/server";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { writeFile, unlink } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { getPhotoCollection } from "@/lib/photo-collection";

const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const nominatimContact =
  process.env.NOMINATIM_CONTACT_EMAIL ??
  process.env.NOMINATIM_CONTACT_API ??
  "hackku-2026@example.com";
const execFileAsync = promisify(execFile);

type ClassifierResult = {
  modelName: string;
  imageSize: number;
  prediction: string;
  confidence: number;
  severity: string;
  recommendedAction: string;
};

type GeminiNarrative = {
  description: string;
  authority: string;
};

function getProjectRoot() {
  const cwd = process.cwd();
  const candidates = [
    cwd,
    path.join(cwd, "web"),
    path.resolve(cwd, ".."),
    path.resolve(cwd, "..", "web"),
  ];

  for (const candidate of new Set(candidates)) {
    if (existsSync(path.join(candidate, "predict.py"))) {
      return candidate;
    }
  }

  return cwd;
}

function getPythonExecutable(projectRoot: string) {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }

  const parent = path.resolve(projectRoot, "..");
  const candidates = [
    path.join(projectRoot, ".venv", "Scripts", "python.exe"),
    path.join(projectRoot, ".venv", "bin", "python"),
    path.join(parent, ".venv", "Scripts", "python.exe"),
    path.join(parent, ".venv", "bin", "python"),
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return process.platform === "win32" ? "python" : "python3";
}

function parseClassifierOutput(stdout: string): ClassifierResult {
  const rows = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const values = new Map<string, string>();
  for (const row of rows) {
    const colonIndex = row.indexOf(":");
    if (colonIndex <= 0) {
      continue;
    }

    const key = row.slice(0, colonIndex).trim();
    const value = row.slice(colonIndex + 1).trim();
    values.set(key, value);
  }

  const prediction = values.get("prediction") ?? "";
  if (!prediction) {
    throw new Error("Classifier output is missing a prediction.");
  }

  const confidenceRaw = Number(values.get("confidence") ?? Number.NaN);
  const confidence = Number.isFinite(confidenceRaw)
    ? Math.max(0, Math.min(1, confidenceRaw))
    : 0;

  return {
    modelName: values.get("model_name") ?? "baseline",
    imageSize: Number(values.get("image_size") ?? 224),
    prediction,
    confidence,
    severity: values.get("severity") ?? "low",
    recommendedAction:
      values.get("recommended_action") ?? "Schedule a visual field inspection.",
  };
}

async function runClassifier(
  imageBuffer: Buffer,
  originalFilename: string,
): Promise<ClassifierResult> {
  const projectRoot = getProjectRoot();
  const modelPath = path.join(projectRoot, "artifacts_new_design", "baseline_model.pt");
  const labelsPath = path.join(projectRoot, "artifacts_new_design", "labels.json");
  const predictScriptPath = path.join(projectRoot, "predict.py");

  if (!existsSync(modelPath) || !existsSync(labelsPath) || !existsSync(predictScriptPath)) {
    throw new Error("Classifier artifacts are missing. Ensure baseline_model.pt, labels.json, and predict.py exist.");
  }

  const extension = path.extname(originalFilename).toLowerCase();
  const safeExtension = [".jpg", ".jpeg", ".png", ".webp"].includes(extension)
    ? extension
    : ".jpg";
  const tempImagePath = path.join(
    os.tmpdir(),
    `roadscout-${Date.now()}-${Math.random().toString(16).slice(2)}${safeExtension}`,
  );

  await writeFile(tempImagePath, imageBuffer);
  const pythonExecutable = getPythonExecutable(projectRoot);

  try {
    const { stdout } = await execFileAsync(
      pythonExecutable,
      [
        predictScriptPath,
        "--image",
        tempImagePath,
        "--model",
        modelPath,
        "--labels",
        labelsPath,
      ],
      {
        cwd: projectRoot,
        timeout: 120_000,
        maxBuffer: 1024 * 1024,
      },
    );

    return parseClassifierOutput(stdout);
  } catch (error) {
    console.error("Local classifier execution failed.", error);
    throw new Error("Local classifier failed. Verify Python environment and model artifacts.");
  } finally {
    await unlink(tempImagePath).catch(() => undefined);
  }
}

async function reverseGeocodeCoordinates(latitude: number, longitude: number) {
  const reverseUrl = new URL("https://nominatim.openstreetmap.org/reverse");
  reverseUrl.searchParams.set("lat", String(latitude));
  reverseUrl.searchParams.set("lon", String(longitude));
  reverseUrl.searchParams.set("format", "jsonv2");
  reverseUrl.searchParams.set("zoom", "18");
  reverseUrl.searchParams.set("addressdetails", "1");

  const response = await fetch(reverseUrl, {
    cache: "no-store",
    headers: {
      "User-Agent": `RoadScout/0.1 (${nominatimContact})`,
      "Accept-Language": "en-US,en;q=0.9",
    },
  });

  if (!response.ok) {
    throw new Error("OpenStreetMap reverse geocoding request failed.");
  }

  const payload = (await response.json()) as {
    display_name?: string;
  };

  if (!payload.display_name) {
    throw new Error("Coordinates could not be reverse geocoded.");
  }

  return {
    formattedAddress: payload.display_name,
    latitude,
    longitude,
  };
}

function defaultAuthorityFromAddress(address: string) {
  const parts = address
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);

  const city = parts.length >= 2 ? parts[1] : parts[0];
  if (!city) {
    return "Your local city public works department or 311 line";
  }

  return `${city} Public Works Department / 311`;
}

function parseGeminiTextToNarrative(
  responseText: string,
  fallbackAuthority: string,
): GeminiNarrative {
  const cleaned = responseText.trim();
  if (!cleaned) {
    throw new Error("Gemini returned an empty response.");
  }

  const codeFence = cleaned.match(/```json\s*([\s\S]*?)\s*```/i);
  const jsonCandidate = codeFence?.[1] ?? cleaned;

  try {
    const parsed = JSON.parse(jsonCandidate) as {
      description?: unknown;
      authority?: unknown;
    };

    const description =
      typeof parsed.description === "string" && parsed.description.trim()
        ? parsed.description.trim()
        : cleaned;
    const authority =
      typeof parsed.authority === "string" && parsed.authority.trim()
        ? parsed.authority.trim()
        : fallbackAuthority;

    return {
      description,
      authority,
    };
  } catch {
    return {
      description: cleaned,
      authority: fallbackAuthority,
    };
  }
}

function getGeminiModelCandidates() {
  const configuredModels = (process.env.GEMINI_MODEL ?? "")
    .split(",")
    .map((modelName) => modelName.trim())
    .filter(Boolean);

  const defaults = ["gemini-2.0-flash", "gemini-1.5-flash"];

  return [...new Set([...configuredModels, ...defaults])];
}

async function createGeminiNarrative(
  classifier: ClassifierResult,
  address: string,
): Promise<GeminiNarrative> {
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GEMINI_API;
  if (!apiKey) {
    throw new Error("Missing GEMINI_API or GEMINI_API_KEY in environment.");
  }

  const modelCandidates = getGeminiModelCandidates();
  const apiVersions = ["v1beta", "v1"] as const;
  const fallbackAuthority = defaultAuthorityFromAddress(address);

  const prompt = [
    "You are assisting with municipal road-damage reporting.",
    "Use the classifier output and location to provide:",
    "1) A brief 2-3 sentence summary for a resident report.",
    "2) The best local authority to report this to.",
    "Return STRICT JSON only in this shape:",
    '{"description":"...","authority":"..."}',
    "Classifier data:",
    `- Predicted class: ${classifier.prediction}`,
    `- Confidence: ${(classifier.confidence * 100).toFixed(1)}%`,
    `- Severity: ${classifier.severity}`,
    `- Recommended action: ${classifier.recommendedAction}`,
    `Location: ${address}`,
    `Fallback authority if uncertain: ${fallbackAuthority}`,
  ].join("\n");

  const requestBody = {
    contents: [
      {
        role: "user",
        parts: [{ text: prompt }],
      },
    ],
    generationConfig: {
      temperature: 0.2,
      responseMimeType: "application/json",
    },
  };

  const notFoundAttempts: string[] = [];

  for (const modelName of modelCandidates) {
    for (const apiVersion of apiVersions) {
      const url = `https://generativelanguage.googleapis.com/${apiVersion}/models/${encodeURIComponent(modelName)}:generateContent?key=${encodeURIComponent(apiKey)}`;

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const responseBody = await response.text();

        if (response.status === 404) {
          notFoundAttempts.push(`${modelName}@${apiVersion}`);
          continue;
        }

        throw new Error(
          `Gemini request failed with status ${response.status} for model ${modelName} (${apiVersion}). ${responseBody}`.trim(),
        );
      }

      const payload = (await response.json()) as {
        candidates?: Array<{
          content?: {
            parts?: Array<{
              text?: string;
            }>;
          };
        }>;
      };

      const text =
        payload.candidates?.[0]?.content?.parts
          ?.map((part) => part.text ?? "")
          .join("\n")
          .trim() ?? "";

      return parseGeminiTextToNarrative(text, fallbackAuthority);
    }
  }

  throw new Error(
    `Gemini model endpoint returned 404 for all candidates: ${notFoundAttempts.join(", ")}. Set GEMINI_MODEL to a valid model for your API key.`,
  );
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const photo = formData.get("photo");
    const latitudeValue = formData.get("latitude");
    const longitudeValue = formData.get("longitude");

    const latitude =
      typeof latitudeValue === "string" ? Number(latitudeValue) : Number.NaN;
    const longitude =
      typeof longitudeValue === "string" ? Number(longitudeValue) : Number.NaN;

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

    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      return NextResponse.json(
        {
          error:
            "Location is required. Allow browser location access before submitting.",
        },
        { status: 400 },
      );
    }

    const geocodedAddress = await reverseGeocodeCoordinates(latitude, longitude);
    const buffer = Buffer.from(await photo.arrayBuffer());
    const classifier = await runClassifier(buffer, photo.name);

    let analysisSummary = "";
    let authorityToReport = defaultAuthorityFromAddress(
      geocodedAddress.formattedAddress,
    );
    let analysisStatus: "complete" | "classifier-only" = "classifier-only";
    let analysisNote = "";

    try {
      const geminiNarrative = await createGeminiNarrative(
        classifier,
        geocodedAddress.formattedAddress,
      );
      analysisSummary = geminiNarrative.description;
      authorityToReport = geminiNarrative.authority;
      analysisStatus = "complete";
    } catch (error) {
      console.error("Gemini narrative generation failed.", error);
      analysisSummary =
        `Detected ${classifier.prediction} with ${(classifier.confidence * 100).toFixed(1)}% confidence. ` +
        `${classifier.recommendedAction}.`;
      analysisNote =
        error instanceof Error
          ? error.message
          : "Gemini analysis unavailable.";
    }

    const collection = await getPhotoCollection();

    const result = await collection.insertOne({
      filename: photo.name,
      contentType: photo.type,
      size: photo.size,
      uploadedAt: new Date(),
      image: buffer,
      analysisStatus,
      analysisNote,
      classifier,
      analysisSummary,
      authorityToReport,
      address: geocodedAddress.formattedAddress,
      location: {
        latitude: geocodedAddress.latitude,
        longitude: geocodedAddress.longitude,
      },
    });

    return NextResponse.json({
      id: result.insertedId.toString(),
      filename: photo.name,
      address: geocodedAddress.formattedAddress,
      classifier,
      analysisSummary,
      authorityToReport,
      analysisStatus,
      analysisNote,
    });
  } catch (error) {
    console.error("Unable to save photo to MongoDB.", error);

    const message =
      error instanceof Error ? error.message : "Unexpected upload error.";

    return NextResponse.json(
      {
        error:
          message === "Coordinates could not be reverse geocoded."
            ? "That location could not be matched to an address. Try submitting again with location services enabled."
            : message === "OpenStreetMap reverse geocoding request failed."
              ? "Location geocoding failed. Verify OpenStreetMap access and contact email settings."
              : message ===
                  "Local classifier failed. Verify Python environment and model artifacts."
                ? "The local AI classifier failed. Check Python/.venv and artifact files before retrying."
            : "The report could not be saved. Verify MongoDB and OpenStreetMap geocoding access.",
      },
      { status: 503 },
    );
  }
}
