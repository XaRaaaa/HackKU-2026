# Next.js Photo Intake Boilerplate

This folder contains a simple Next.js app that:

- uploads photos from the browser
- stores image bytes and metadata in MongoDB
- lists recent uploads on the homepage
- leaves a clean `analysisStatus` field to plug your ML step in later

## Quick start

1. Install dependencies:

```bash
npm install
```

2. Copy `.env.example` to `.env.local` and fill in your MongoDB values.

3. Run the app:

```bash
npm run dev
```

4. Open `http://localhost:3000`

## Project structure

- `app/page.tsx`: homepage with upload UI and recent photos
- `app/api/photos/route.ts`: upload endpoint
- `app/api/photos/[id]/route.ts`: image-serving endpoint
- `lib/mongodb.ts`: MongoDB connection helper

## Notes

- This is an MVP starter, not production-grade object storage.
- For larger scale, move image blobs to S3, Cloudinary, or Mongo GridFS.
- The current schema already includes `analysisStatus` so you can add model results next.
