import { MongoClient } from "mongodb";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

type EnvLike = Record<string, string | undefined>;

function parseDotEnvFile(filePath: string): EnvLike {
  if (!existsSync(filePath)) {
    return {};
  }

  const raw = readFileSync(filePath, "utf8");
  const env: EnvLike = {};

  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex <= 0) {
      continue;
    }

    const key = trimmed.slice(0, equalsIndex).trim();
    let value = trimmed.slice(equalsIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    env[key] = value;
  }

  return env;
}

function getFallbackEnv(): EnvLike {
  const cwd = process.cwd();
  const candidates = [
    path.join(cwd, ".env.local"),
    path.join(cwd, ".env"),
    path.join(cwd, "..", ".env.local"),
    path.join(cwd, "..", ".env"),
  ];

  for (const filePath of candidates) {
    const parsed = parseDotEnvFile(filePath);
    if (Object.keys(parsed).length > 0) {
      return parsed;
    }
  }

  return {};
}

function getEnvValue(name: string, fallbackEnv: EnvLike) {
  const value = process.env[name] ?? fallbackEnv[name];
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function getMongoConfig() {
  const fallbackEnv = getFallbackEnv();
  const mongoUri =
    getEnvValue("MONGODB_URI", fallbackEnv) ??
    getEnvValue("MONGO_URI", fallbackEnv);
  const mongoDatabaseName =
    getEnvValue("MONGODB_DB", fallbackEnv) ??
    getEnvValue("MONGO_DB_NAME", fallbackEnv) ??
    getEnvValue("MONGODB_DATABASE", fallbackEnv);

  if (!mongoUri) {
    throw new Error(
      "Missing MongoDB URI. Set MONGODB_URI (or MONGO_URI) in web/.env.local for local runs, or in Vercel project environment variables.",
    );
  }

  if (!mongoDatabaseName) {
    throw new Error(
      "Missing MongoDB database name. Set MONGODB_DB (or MONGO_DB_NAME / MONGODB_DATABASE) in web/.env.local or Vercel environment variables.",
    );
  }

  if (!/^mongodb(\+srv)?:\/\//i.test(mongoUri)) {
    throw new Error(
      "Invalid MongoDB URI format. It must start with mongodb:// or mongodb+srv://.",
    );
  }

  return {
    mongoUri,
    mongoDatabaseName,
  };
}

function getClientPromise(mongoUri: string) {
  if (!global._mongoClientPromise) {
    const client = new MongoClient(mongoUri, {
      serverSelectionTimeoutMS: 5000,
    });
    global._mongoClientPromise = client.connect();
  }

  return global._mongoClientPromise;
}

export async function getDatabase() {
  const { mongoUri, mongoDatabaseName } = getMongoConfig();

  try {
    const connectedClient = await getClientPromise(mongoUri);
    return connectedClient.db(mongoDatabaseName);
  } catch (error) {
    global._mongoClientPromise = undefined;
    throw error;
  }
}
