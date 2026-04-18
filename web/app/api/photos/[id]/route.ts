import { ObjectId } from "mongodb";
import { getPhotoCollection } from "@/lib/photo-collection";

export async function GET(
  _request: Request,
  { params }: { params: { id: string } },
) {
  if (!ObjectId.isValid(params.id)) {
    return new Response("Invalid photo id.", { status: 400 });
  }

  const collection = await getPhotoCollection();
  const photo = await collection.findOne(
    { _id: new ObjectId(params.id) },
    {
      projection: {
        image: 1,
        contentType: 1,
      },
    },
  );

  if (!photo?.image || !photo?.contentType) {
    return new Response("Photo not found.", { status: 404 });
  }

  const image =
    photo.image instanceof Buffer ? photo.image : Buffer.from(photo.image.buffer);

  return new Response(image, {
    headers: {
      "Content-Type": String(photo.contentType),
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
