import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_DB;

if (!uri) {
  throw new Error("Missing MONGODB_URI in environment.");
}

if (!dbName) {
  throw new Error("Missing MONGODB_DB in environment.");
}

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

const client = new MongoClient(uri);
const clientPromise = global._mongoClientPromise ?? client.connect();

if (process.env.NODE_ENV !== "production") {
  global._mongoClientPromise = clientPromise;
}

export async function getDatabase() {
  const connectedClient = await clientPromise;
  console.log("Connected to DB:", dbName);
  return connectedClient.db(dbName);
}
