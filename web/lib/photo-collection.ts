import { getDatabase } from "@/lib/mongodb";

export type StoredPhoto = {
  _id: string;
  filename: string;
  contentType: string;
  uploadedAt: string;
  size: number;
  address?: string;
  location?: {
    latitude: number;
    longitude: number;
  };
};

export async function getPhotoCollection() {
  const db = await getDatabase();
  return db.collection("photos");
}
