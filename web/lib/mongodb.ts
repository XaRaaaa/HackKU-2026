import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_DB;

if (!uri) {
  throw new Error("Missing MONGODB_URI in environment.");
}

if (!dbName) {
  throw new Error("Missing MONGODB_DB in environment.");
}

const mongoUri = uri;
const mongoDatabaseName = dbName;

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

function getClientPromise() {
  if (!global._mongoClientPromise) {
    const client = new MongoClient(mongoUri, {
      serverSelectionTimeoutMS: 5000,
    });
    global._mongoClientPromise = client.connect();
  }

  return global._mongoClientPromise;
}

export async function getDatabase() {
  try {
    const connectedClient = await getClientPromise();
    return connectedClient.db(mongoDatabaseName);
  } catch (error) {
    global._mongoClientPromise = undefined;
    throw error;
  }
}
