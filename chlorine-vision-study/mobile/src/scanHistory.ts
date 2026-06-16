import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Location from 'expo-location';

export type ScanRecord = {
  id: string;
  concentration: number;
  unit: string;
  timestamp: string;
  locationLabel: string;
  latitude: number | null;
  longitude: number | null;
};

const STORAGE_KEY = 'chlorine_scan_history_v1';
const MAX_HISTORY_ITEMS = 50;
const LOCATION_TIMEOUT_MS = 3500;

function isValidScanRecord(value: unknown): value is ScanRecord {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;
  return (
    typeof record.id === 'string' &&
    typeof record.concentration === 'number' &&
    typeof record.unit === 'string' &&
    typeof record.timestamp === 'string' &&
    typeof record.locationLabel === 'string'
  );
}

function sortNewestFirst(records: ScanRecord[]) {
  return [...records].sort(
    (left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime()
  );
}

export async function loadScanHistory() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return sortNewestFirst(parsed.filter(isValidScanRecord)).slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

export async function saveScanRecord(record: ScanRecord) {
  const existing = await loadScanHistory();
  const updated = sortNewestFirst([record, ...existing]).slice(0, MAX_HISTORY_ITEMS);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

export async function deleteScanRecord(recordId: string) {
  const existing = await loadScanHistory();
  const updated = existing.filter((record) => record.id !== recordId);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

function formatCoordinateLocation(latitude: number, longitude: number) {
  return `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
}

function buildLocationLabel(
  places: Location.LocationGeocodedAddress[],
  latitude: number,
  longitude: number
) {
  const primary = places[0];
  if (!primary) {
    return formatCoordinateLocation(latitude, longitude);
  }

  const localityParts = [
    primary.name,
    primary.district,
    primary.city,
    primary.region,
  ].filter((part, index, all) => typeof part === 'string' && all.indexOf(part) === index);

  if (!localityParts.length) {
    return formatCoordinateLocation(latitude, longitude);
  }

  return localityParts.join(', ');
}

async function resolveCurrentScanLocation() {
  try {
    const permission = await Location.requestForegroundPermissionsAsync();
    if (!permission.granted) {
      return {
        locationLabel: 'Location not shared',
        latitude: null,
        longitude: null,
      };
    }

    const lastKnown = await Location.getLastKnownPositionAsync();
    const position =
      lastKnown ||
      (await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      }));

    if (!position) {
      return {
        locationLabel: 'Location unavailable',
        latitude: null,
        longitude: null,
      };
    }

    const latitude = position.coords.latitude;
    const longitude = position.coords.longitude;

    try {
      const places = await Location.reverseGeocodeAsync({ latitude, longitude });
      return {
        locationLabel: buildLocationLabel(places, latitude, longitude),
        latitude,
        longitude,
      };
    } catch {
      return {
        locationLabel: formatCoordinateLocation(latitude, longitude),
        latitude,
        longitude,
      };
    }
  } catch {
    return {
      locationLabel: 'Location unavailable',
      latitude: null,
      longitude: null,
    };
  }
}

export async function getCurrentScanLocation() {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  try {
    return await Promise.race([
      resolveCurrentScanLocation(),
      new Promise<{
        locationLabel: string;
        latitude: null;
        longitude: null;
      }>((resolve) => {
        timeoutId = setTimeout(() => {
          resolve({
            locationLabel: 'Location unavailable',
            latitude: null,
            longitude: null,
          });
        }, LOCATION_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}
