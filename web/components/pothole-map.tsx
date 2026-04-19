"use client";

import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import { divIcon, latLngBounds } from "leaflet";

type PotholeMapItem = {
  id: string;
  title: string;
  uploadedAt: string;
  address?: string;
  latitude: number;
  longitude: number;
};

type PotholeMapProps = {
  items: PotholeMapItem[];
};

function FitBounds({ items }: { items: PotholeMapItem[] }) {
  const map = useMap();

  useEffect(() => {
    if (items.length === 0) {
      map.setView([38.9717, -95.2353], 13);
      return;
    }

    if (items.length === 1) {
      map.setView([items[0].latitude, items[0].longitude], 15);
      return;
    }

    const bounds = latLngBounds(
      items.map((item) => [item.latitude, item.longitude] as [number, number]),
    );
    map.fitBounds(bounds, { padding: [40, 40] });
  }, [items, map]);

  return null;
}

const potholeIcon = divIcon({
  className: "pothole-marker",
  html: '<span style="display:block;width:18px;height:18px;border-radius:999px;background:#2367c8;border:3px solid #f3f8ff;box-shadow:0 8px 20px rgba(24,64,121,.28);"></span>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

export function PotholeMap({ items }: PotholeMapProps) {
  const [mapLoaded, setMapLoaded] = useState(false);
  const center = useMemo<[number, number]>(() => {
    if (items.length > 0) {
      return [items[0].latitude, items[0].longitude];
    }

    return [38.9717, -95.2353];
  }, [items]);

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div
        style={{
          width: "100%",
          minHeight: 360,
          borderRadius: 26,
          overflow: "hidden",
          border: "1px solid var(--card-border)",
          background:
            "linear-gradient(135deg, rgba(210, 228, 252, 0.95), rgba(236, 245, 255, 0.98))",
        }}
      >
        <MapContainer
          center={center}
          zoom={13}
          scrollWheelZoom
          style={{ height: "360px", width: "100%" }}
          whenReady={() => setMapLoaded(true)}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds items={items} />
          {items.map((item) => (
            <Marker
              key={item.id}
              position={[item.latitude, item.longitude]}
              icon={potholeIcon}
            >
              <Popup>
                <div style={{ fontFamily: "Georgia, serif", lineHeight: 1.5 }}>
                  <strong>{item.title}</strong>
                  <br />
                  {item.address ?? "Address unavailable"}
                  <br />
                  Logged {item.uploadedAt}
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
      {!mapLoaded ? (
        <p
          style={{
            margin: 0,
            color: "var(--muted)",
            lineHeight: 1.6,
          }}
        >
          Loading OpenStreetMap...
        </p>
      ) : items.length === 0 ? (
        <p
          style={{
            margin: 0,
            color: "var(--muted)",
            lineHeight: 1.6,
          }}
        >
          The base map is live. Add reports with addresses to place pothole pins.
        </p>
      ) : null}
    </div>
  );
}
