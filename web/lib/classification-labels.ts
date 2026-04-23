const KNOWN_LABELS: Record<string, string> = {
  D00_longitudinal_crack: "Longitudinal Crack (D00)",
  D10_transverse_crack: "Transverse Crack (D10)",
  D20_alligator_crack: "Alligator Crack (D20)",
  D40_pothole: "Pothole (D40)",
  D43_crosswalk_blur: "Crosswalk Blur (D43)",
  no_damage: "No Visible Damage",
};

function titleCase(value: string) {
  return value
    .split(" ")
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

export function formatClassifierLabel(rawLabel: string) {
  const cleaned = rawLabel.trim();
  if (!cleaned) {
    return "Unknown";
  }

  const known = KNOWN_LABELS[cleaned];
  if (known) {
    return known;
  }

  const codedMatch = cleaned.match(/^([A-Za-z]\d{2})[_-](.+)$/);
  if (codedMatch) {
    const [, code, rest] = codedMatch;
    return `${titleCase(rest.replaceAll("_", " ").replaceAll("-", " "))} (${code.toUpperCase()})`;
  }

  return titleCase(cleaned.replaceAll("_", " ").replaceAll("-", " "));
}

export function formatSeverityLabel(rawSeverity: string) {
  const severity = rawSeverity.trim().toLowerCase();
  if (!severity) {
    return "Unknown";
  }

  if (severity === "low" || severity === "medium" || severity === "high") {
    return severity[0].toUpperCase() + severity.slice(1);
  }

  return titleCase(severity.replaceAll("_", " ").replaceAll("-", " "));
}
