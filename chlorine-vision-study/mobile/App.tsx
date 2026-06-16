import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  LayoutChangeEvent,
  PanResponder,
  Platform,
  Pressable,
  ScrollView,
  Text,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import {
  analyzeSample,
  checkAnalyzerReady,
  getAnalyzerUnavailableMessage,
} from './src/analyzer';
import {
  deleteScanRecord,
  getCurrentScanLocation,
  loadScanHistory,
  saveScanRecord,
  type ScanRecord,
} from './src/scanHistory';
import { styles } from './src/styles';

type Screen = 'home' | 'capture' | 'camera' | 'crop' | 'processing' | 'result' | 'history';
type AnalyzerStatus = 'checking' | 'ready' | 'unavailable';
type HistoryOrigin = 'home' | 'result';

type CropRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type SelectedPhoto = {
  name: string;
  type: string;
  originalUri: string;
  uploadUri?: string;
  uploadBase64?: string;
  width: number;
  height: number;
  cropRect: CropRect;
};

type PredictionResult = {
  concentration: number;
  isSafe: boolean;
  modelName: string;
  timestamp: string;
  unit: string;
};

const SAFE_MIN = 0.2;
const SAFE_MAX = 2.0;
const MIN_EXPORTED_CROP = 0.04;
const FRAME_HEIGHT_RATIO = 0.7;
const FRAME_WIDTH_TO_HEIGHT = 0.42;
const MAX_FRAME_SCALE = 4;
const ZOOM_STEP = 0.2;
const DEFAULT_CROP_RECT: CropRect = {
  x: 0.28,
  y: 0.18,
  width: 0.44,
  height: 0.58,
};

function buildResult(
  predictedConcentration: number,
  unit: string,
  modelName: string,
  timestamp: string
): PredictionResult {
  return {
    concentration: predictedConcentration,
    isSafe: predictedConcentration >= SAFE_MIN && predictedConcentration <= SAFE_MAX,
    modelName,
    timestamp,
    unit,
  };
}

function describeConcentration(concentration: number) {
  if (concentration < SAFE_MIN) {
    return {
      badge: 'Below target range',
      tone: 'warning' as const,
      copy: 'The sample looks under-chlorinated and may not be disinfected enough.',
    };
  }

  if (concentration > SAFE_MAX) {
    return {
      badge: 'Above target range',
      tone: 'warning' as const,
      copy: 'The sample looks over-chlorinated and may cause taste or odor issues.',
    };
  }

  return {
    badge: 'Within target range',
    tone: 'safe' as const,
    copy: 'The sample falls within the recommended chlorine range for drinking water.',
  };
}

function describeReading(result: PredictionResult) {
  return describeConcentration(result.concentration);
}

function clampUnitRect(rect: CropRect): CropRect {
  const x = Math.min(Math.max(rect.x, 0), 1);
  const y = Math.min(Math.max(rect.y, 0), 1);
  const width = Math.min(Math.max(rect.width, MIN_EXPORTED_CROP), 1 - x);
  const height = Math.min(Math.max(rect.height, MIN_EXPORTED_CROP), 1 - y);
  return { x, y, width, height };
}

function computeDisplayedImageRect(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number
) {
  if (!containerWidth || !containerHeight || !imageWidth || !imageHeight) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  const imageAspect = imageWidth / imageHeight;
  const containerAspect = containerWidth / containerHeight;

  if (imageAspect > containerAspect) {
    const width = containerWidth;
    const height = width / imageAspect;
    return { x: 0, y: (containerHeight - height) / 2, width, height };
  }

  const height = containerHeight;
  const width = height * imageAspect;
  return { x: (containerWidth - width) / 2, y: 0, width, height };
}

function computeFramingBox(containerWidth: number, containerHeight: number) {
  if (!containerWidth || !containerHeight) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  const height = Math.min(containerHeight * FRAME_HEIGHT_RATIO, containerWidth * 1.3);
  const width = height * FRAME_WIDTH_TO_HEIGHT;

  return {
    x: (containerWidth - width) / 2,
    y: (containerHeight - height) / 2,
    width,
    height,
  };
}

function computeTransformedImageRect(
  baseRect: { x: number; y: number; width: number; height: number },
  scale: number,
  offsetX: number,
  offsetY: number
) {
  const width = baseRect.width * scale;
  const height = baseRect.height * scale;
  const x = baseRect.x + (baseRect.width - width) / 2 + offsetX;
  const y = baseRect.y + (baseRect.height - height) / 2 + offsetY;
  return { x, y, width, height };
}

function clampFrameTransform(
  baseRect: { x: number; y: number; width: number; height: number },
  frameRect: { x: number; y: number; width: number; height: number },
  scale: number,
  offsetX: number,
  offsetY: number
) {
  if (!baseRect.width || !baseRect.height || !frameRect.width || !frameRect.height) {
    return { scale: 1, offsetX: 0, offsetY: 0 };
  }

  const minScale = Math.max(1, frameRect.width / baseRect.width, frameRect.height / baseRect.height);
  const nextScale = Math.min(Math.max(scale, minScale), MAX_FRAME_SCALE);
  const transformedRect = computeTransformedImageRect(baseRect, nextScale, 0, 0);

  const minOffsetX = frameRect.x + frameRect.width - transformedRect.x - transformedRect.width;
  const maxOffsetX = frameRect.x - transformedRect.x;
  const minOffsetY = frameRect.y + frameRect.height - transformedRect.y - transformedRect.height;
  const maxOffsetY = frameRect.y - transformedRect.y;

  return {
    scale: nextScale,
    offsetX: Math.min(Math.max(offsetX, minOffsetX), maxOffsetX),
    offsetY: Math.min(Math.max(offsetY, minOffsetY), maxOffsetY),
  };
}

function computeCropRectFromFrame(
  photo: SelectedPhoto,
  previewWidth: number,
  previewHeight: number,
  scale: number,
  offsetX: number,
  offsetY: number
) {
  const baseRect = computeDisplayedImageRect(previewWidth, previewHeight, photo.width, photo.height);
  const frameRect = computeFramingBox(previewWidth, previewHeight);
  const transformedRect = computeTransformedImageRect(baseRect, scale, offsetX, offsetY);

  return clampUnitRect({
    x: (frameRect.x - transformedRect.x) / transformedRect.width,
    y: (frameRect.y - transformedRect.y) / transformedRect.height,
    width: frameRect.width / transformedRect.width,
    height: frameRect.height / transformedRect.height,
  });
}

function toJpegName(name: string) {
  return name.replace(/\.[^.]+$/, '.jpg');
}

function formatScanDate(timestamp: string) {
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function formatScanTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  });
}

export default function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [analyzerStatus, setAnalyzerStatus] = useState<AnalyzerStatus>('checking');
  const [selectedPhoto, setSelectedPhoto] = useState<SelectedPhoto | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [currentScan, setCurrentScan] = useState<ScanRecord | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanRecord[]>([]);
  const [historyOrigin, setHistoryOrigin] = useState<HistoryOrigin>('home');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    verifyAnalyzer();
    hydrateScanHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function hydrateScanHistory() {
    const storedHistory = await loadScanHistory();
    setScanHistory(storedHistory);
  }

  async function verifyAnalyzer() {
    setAnalyzerStatus('checking');

    try {
      await checkAnalyzerReady();
      setAnalyzerStatus('ready');
      setErrorMessage('');
    } catch {
      setAnalyzerStatus('unavailable');
    }
  }

  async function persistScanResult(nextResult: PredictionResult, timestamp: string) {
    try {
      const location = await getCurrentScanLocation();
      const savedScan: ScanRecord = {
        id: `${timestamp}-${Date.now()}`,
        concentration: nextResult.concentration,
        unit: nextResult.unit,
        timestamp,
        locationLabel: location.locationLabel,
        latitude: location.latitude,
        longitude: location.longitude,
      };
      const updatedHistory = await saveScanRecord(savedScan);

      setCurrentScan(savedScan);
      setScanHistory(updatedHistory);
    } catch {
      setCurrentScan(null);
    }
  }

  async function handleDeleteScan(recordId: string) {
    const updatedHistory = await deleteScanRecord(recordId);
    setScanHistory(updatedHistory);
    if (currentScan?.id === recordId) {
      setCurrentScan(null);
    }
  }

  function openHistory(origin: HistoryOrigin) {
    setHistoryOrigin(origin);
    setScreen('history');
  }

  async function analyzePreparedPhoto(photo: SelectedPhoto, nextCropRect?: CropRect) {
    setErrorMessage('');
    setScreen('processing');

    try {
      const photoForAnalysis = nextCropRect
        ? {
            ...photo,
            cropRect: nextCropRect,
            uploadUri: undefined,
            uploadBase64: undefined,
          }
        : photo;
      const croppedPhoto = await createCroppedPhoto(photoForAnalysis);
      setSelectedPhoto(croppedPhoto);
      const payload = await analyzeSample({
        base64: croppedPhoto.uploadBase64 || '',
      });
      const timestamp = payload.timestamp || new Date().toISOString();
      const nextResult = buildResult(
        Number(payload.predicted_concentration),
        payload.unit || 'mg/L',
        payload.model_name || 'Unknown',
        timestamp
      );
      setResult(nextResult);
      setScreen('result');
      void persistScanResult(nextResult, timestamp);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : getAnalyzerUnavailableMessage();
      setErrorMessage(message);
      setScreen('capture');
    }
  }

  async function handleImagePick(mode: 'library') {
    setErrorMessage('');

    try {
      let pickerResult: ImagePicker.ImagePickerResult;

      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Photo library access needed', 'Please allow photo access to upload an existing image.');
        return;
      }

      pickerResult = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 1,
        allowsEditing: true,
        aspect: [3, 5],
      });

      if (pickerResult.canceled || !pickerResult.assets.length) {
        return;
      }

      const asset = pickerResult.assets[0];
      const fileExtension = asset.fileName?.split('.').pop()?.toLowerCase();
      const mimeType = asset.mimeType?.toLowerCase();
      const normalizedExtension =
        fileExtension ||
        (mimeType === 'image/png'
          ? 'png'
          : mimeType === 'image/heif'
            ? 'heif'
            : mimeType === 'image/heic'
              ? 'heic'
              : 'jpg');
      const fileType =
        mimeType ||
        (normalizedExtension === 'png'
          ? 'image/png'
          : normalizedExtension === 'heif'
            ? 'image/heif'
            : normalizedExtension === 'heic'
              ? 'image/heic'
              : 'image/jpeg');
      const fileName = asset.fileName || `sample-${Date.now()}.${normalizedExtension}`;

      const preparedPhoto = {
        name: fileName,
        type: fileType,
        originalUri: asset.uri,
        width: asset.width || 1,
        height: asset.height || 1,
        cropRect: { x: 0, y: 0, width: 1, height: 1 },
      };
      setSelectedPhoto(preparedPhoto);
      await analyzePreparedPhoto(preparedPhoto);
    } catch {
      setErrorMessage('Could not open the image picker. Please try again.');
    }
  }

  async function handleCapturedCameraPhoto(photo: {
    uri: string;
    width?: number;
    height?: number;
  }) {
    const preparedPhoto = {
      name: `camera-sample-${Date.now()}.jpg`,
      type: 'image/jpeg',
      originalUri: photo.uri,
      width: photo.width || 1,
      height: photo.height || 1,
      cropRect: { x: 0, y: 0, width: 1, height: 1 },
    };
    setSelectedPhoto(preparedPhoto);
    await analyzePreparedPhoto(preparedPhoto);
  }

  async function createCroppedPhoto(photo: SelectedPhoto) {
    const crop = {
      originX: Math.max(0, Math.round(photo.cropRect.x * photo.width)),
      originY: Math.max(0, Math.round(photo.cropRect.y * photo.height)),
      width: Math.max(1, Math.round(photo.cropRect.width * photo.width)),
      height: Math.max(1, Math.round(photo.cropRect.height * photo.height)),
    };

    const resultImage = await ImageManipulator.manipulateAsync(
      photo.originalUri,
      [{ crop }],
      { base64: true, compress: 1, format: ImageManipulator.SaveFormat.JPEG }
    );

    return {
      ...photo,
      uploadUri: resultImage.uri,
      uploadBase64: resultImage.base64 || undefined,
      name: toJpegName(photo.name),
      type: 'image/jpeg',
    };
  }

  async function analyzePhoto(nextCropRect?: CropRect) {
    if (!selectedPhoto) {
      setErrorMessage('Choose or capture a test-tube image first.');
      return;
    }
    await analyzePreparedPhoto(selectedPhoto, nextCropRect);
  }

  function resetForNewScan() {
    setSelectedPhoto(null);
    setResult(null);
    setCurrentScan(null);
    setErrorMessage('');
    setScreen('capture');
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
        <StatusBar style="light" />
        <KeyboardAvoidingView
          behavior={Platform.select({ ios: 'padding', default: undefined })}
          style={styles.keyboardShell}
        >
          {screen === 'home' && (
            <HomeScreen
              analyzerStatus={analyzerStatus}
              scanHistory={scanHistory}
              onOpenHistory={() => openHistory('home')}
              onRetryConnection={verifyAnalyzer}
              onStart={() => setScreen('capture')}
            />
          )}

          {screen === 'capture' && (
            <CaptureScreen
              errorMessage={errorMessage}
              onBack={() => setScreen('home')}
              onChooseExisting={() => handleImagePick('library')}
              onTakeNew={() => setScreen('camera')}
            />
          )}

          {screen === 'camera' && (
            <GuidedCameraScreen
              onBack={() => setScreen('capture')}
              onCaptured={handleCapturedCameraPhoto}
            />
          )}

          {screen === 'crop' && selectedPhoto && (
            <CropScreen
              errorMessage={errorMessage}
              photo={selectedPhoto}
              onBack={() => setScreen('capture')}
              onAnalyze={analyzePhoto}
            />
          )}

          {screen === 'processing' && <ProcessingScreen />}

          {screen === 'result' && result && (
            <ResultScreen
              currentScan={currentScan}
              photo={selectedPhoto}
              result={result}
              scanHistory={scanHistory}
              onNewScan={resetForNewScan}
              onOpenHistory={() => openHistory('result')}
              onRetake={() => {
                setCurrentScan(null);
                setResult(null);
                setScreen('home');
              }}
            />
          )}

          {screen === 'history' && (
            <HistoryScreen
              scans={scanHistory}
              onBack={() => setScreen(historyOrigin)}
              onDelete={handleDeleteScan}
            />
          )}
        </KeyboardAvoidingView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

function HomeScreen({
  analyzerStatus,
  scanHistory,
  onOpenHistory,
  onRetryConnection,
  onStart,
}: {
  analyzerStatus: AnalyzerStatus;
  scanHistory: ScanRecord[];
  onOpenHistory: () => void;
  onRetryConnection: () => void;
  onStart: () => void;
}) {
  const isReady = analyzerStatus === 'ready';

  return (
    <ScrollView contentContainerStyle={[styles.screenScroll, styles.screenScrollGrow]}>
      <View style={styles.homeIntro}>
        <Text style={styles.homeIntroTitle}>Check chlorine concentration from a photo.</Text>
      </View>

      <View style={styles.panelCard}>
        <Text style={styles.sectionTitle}>Before You Start</Text>
        <View style={styles.instructionsCard}>
          <View style={styles.instructionRow}>
            <View style={styles.instructionNumber}>
              <Text style={styles.instructionNumberText}>1</Text>
            </View>
            <Text style={styles.instructionText}>
              Fill the test tube or 10 mL cell with your water sample.
            </Text>
          </View>
          <View style={styles.instructionRow}>
            <View style={styles.instructionNumber}>
              <Text style={styles.instructionNumberText}>2</Text>
            </View>
            <Text style={styles.instructionText}>
              Add one DPD powder pillow and gently mix until it fully dissolves.
            </Text>
          </View>
          <View style={styles.instructionRow}>
            <View style={styles.instructionNumber}>
              <Text style={styles.instructionNumberText}>3</Text>
            </View>
            <Text style={styles.instructionText}>
              Wait for the sample to turn pink, then place it in good lighting against a plain background.
            </Text>
          </View>
          <View style={styles.instructionRow}>
            <View style={styles.instructionNumber}>
              <Text style={styles.instructionNumberText}>4</Text>
            </View>
            <Text style={styles.instructionText}>
              Take or upload a photo and crop so the pink liquid and test tube are clearly inside the frame.
            </Text>
          </View>
        </View>
        {!isReady && (
          <Pressable
            style={[styles.secondaryButton, analyzerStatus === 'checking' && styles.disabledButton]}
            disabled={analyzerStatus === 'checking'}
            onPress={onRetryConnection}
          >
            <Text style={styles.secondaryButtonText}>
              {analyzerStatus === 'checking' ? 'Loading...' : 'Retry'}
            </Text>
          </Pressable>
        )}
      </View>

      {scanHistory.length > 0 && (
        <HistoryCard
          title="Recent Scans"
          subtitle="Saved on this phone"
          scans={scanHistory.slice(0, 3)}
        />
      )}

      <Pressable style={styles.secondaryButton} onPress={onOpenHistory}>
        <Text style={styles.secondaryButtonText}>View Previous Scans</Text>
      </Pressable>

      <Pressable style={[styles.primaryButton, styles.buttonShadow]} onPress={onStart}>
        <Text style={styles.primaryButtonText}>Start Scan</Text>
      </Pressable>
    </ScrollView>
  );
}

function CaptureScreen({
  errorMessage,
  onBack,
  onChooseExisting,
  onTakeNew,
}: {
  errorMessage: string;
  onBack: () => void;
  onChooseExisting: () => void;
  onTakeNew: () => void;
}) {
  return (
    <ScrollView contentContainerStyle={[styles.screenScroll, styles.screenScrollGrow]}>
      <View style={styles.navRow}>
        <Pressable onPress={onBack}>
          <Text style={styles.backLink}>Back</Text>
        </Pressable>
      </View>

      <View style={styles.panelCard}>
        <Text style={styles.sectionTitle}>Add Sample Photo</Text>
        <Text style={styles.subtleCopy}>Pick a clear image with the pink test tube visible.</Text>

        <View style={styles.previewPlaceholder}>
          <View style={styles.tubeOutline}>
            <View style={styles.tubeLine} />
          </View>
          <Text style={styles.previewPlaceholderTitle}>Keep the tube centered</Text>
          <Text style={styles.previewPlaceholderCopy}>Plain background. Low glare. Pink sample visible.</Text>
        </View>

        <View style={styles.helperBanner}>
          <Text style={styles.helperBannerText}>
            After you choose a photo, the phone will open its built-in crop tool.
          </Text>
        </View>

        <View style={styles.captureOptions}>
          <Pressable style={styles.captureOptionCard} onPress={onChooseExisting}>
            <Text style={styles.captureOptionEyebrow}>Library</Text>
            <Text style={styles.captureOptionTitle}>Upload Photo</Text>
            <Text style={styles.captureOptionCopy}>Use an image already on your phone.</Text>
          </Pressable>
          <Pressable style={[styles.captureOptionCard, styles.captureOptionPrimary]} onPress={onTakeNew}>
            <Text style={[styles.captureOptionEyebrow, styles.captureOptionEyebrowPrimary]}>Camera</Text>
            <Text style={[styles.captureOptionTitle, styles.captureOptionTitlePrimary]}>Take Photo</Text>
            <Text style={[styles.captureOptionCopy, styles.captureOptionCopyPrimary]}>
              Capture a fresh sample image now.
            </Text>
          </Pressable>
        </View>

        {!!errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}
      </View>
    </ScrollView>
  );
}

function GuidedCameraScreen({
  onBack,
  onCaptured,
}: {
  onBack: () => void;
  onCaptured: (photo: { uri: string; width?: number; height?: number }) => Promise<void>;
}) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCapturing, setIsCapturing] = useState(false);
  const cameraRef = useRef<CameraView | null>(null);

  async function handleCapture() {
    if (!cameraRef.current || isCapturing) {
      return;
    }

    try {
      setIsCapturing(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
      });
      if (!photo?.uri) {
        throw new Error('The camera did not return an image.');
      }
      await onCaptured(photo);
    } catch (error) {
      Alert.alert(
        'Photo not captured',
        error instanceof Error ? error.message : 'Please try taking the photo again.'
      );
    } finally {
      setIsCapturing(false);
    }
  }

  if (!permission) {
    return (
      <View style={styles.cameraLoadingScreen}>
        <ActivityIndicator size="large" color="#fff3d5" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.screenStatic}>
        <View style={styles.navRow}>
          <Pressable onPress={onBack}>
            <Text style={styles.backLink}>Back</Text>
          </Pressable>
        </View>

        <View style={styles.panelCard}>
          <Text style={styles.sectionTitle}>Camera Access Needed</Text>
          <Text style={styles.subtleCopy}>
            Allow camera access so the app can show the live guide and help line up the test tube.
          </Text>
          <Pressable style={styles.primaryButton} onPress={requestPermission}>
            <Text style={styles.primaryButtonText}>Allow Camera</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.cameraScreen}>
      <CameraView ref={cameraRef} style={styles.cameraPreview} facing="back">
        <View style={styles.cameraHeader}>
          <Pressable style={styles.cameraBackButton} onPress={onBack}>
            <Text style={styles.cameraBackButtonText}>Back</Text>
          </Pressable>
          <View style={styles.cameraHeaderPill}>
            <Text style={styles.cameraHeaderPillText}>Fit the tube in the guide</Text>
          </View>
        </View>

        <View style={styles.cameraGuideArea} pointerEvents="none">
          <View style={styles.cameraGuideShadeTop} />
          <View style={styles.cameraGuideMiddleRow}>
            <View style={styles.cameraGuideShadeSide} />
            <View style={styles.cameraGuideBox}>
              <View style={styles.cameraGuideInnerLine} />
            </View>
            <View style={styles.cameraGuideShadeSide} />
          </View>
          <View style={styles.cameraGuideShadeBottom} />
        </View>

        <View style={styles.cameraFooter}>
          <Text style={styles.cameraHintText}>
            Keep the full test tube inside the outline and the pink liquid centered.
          </Text>
          <Pressable style={styles.cameraCaptureButton} onPress={handleCapture}>
            <View style={styles.cameraCaptureButtonInner} />
          </Pressable>
          <Text style={styles.cameraHintSubtext}>
            {isCapturing ? 'Capturing photo...' : 'Tap the shutter when the tube is lined up.'}
          </Text>
        </View>
      </CameraView>
    </View>
  );
}

function CropScreen({
  errorMessage,
  photo,
  onBack,
  onAnalyze,
}: {
  errorMessage: string;
  photo: SelectedPhoto;
  onBack: () => void;
  onAnalyze: (cropRect: CropRect) => void;
}) {
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });

  const baseImageRect = useMemo(
    () =>
      computeDisplayedImageRect(previewSize.width, previewSize.height, photo.width, photo.height),
    [photo.height, photo.width, previewSize.height, previewSize.width]
  );

  const frameRect = useMemo(
    () => computeFramingBox(previewSize.width, previewSize.height),
    [previewSize.height, previewSize.width]
  );

  useEffect(() => {
    if (!baseImageRect.width || !baseImageRect.height || !frameRect.width || !frameRect.height) {
      return;
    }

    const next = clampFrameTransform(baseImageRect, frameRect, 1, 0, 0);
    setScale(next.scale);
    setOffset({ x: next.offsetX, y: next.offsetY });
  }, [baseImageRect.height, baseImageRect.width, frameRect.height, frameRect.width, photo.originalUri]);

  const transformedImageRect = useMemo(
    () => computeTransformedImageRect(baseImageRect, scale, offset.x, offset.y),
    [baseImageRect, offset.x, offset.y, scale]
  );

  const frameEdge = {
    left: frameRect.x + frameRect.width,
    top: frameRect.y + frameRect.height,
  };

  function adjustZoom(delta: number) {
    if (!baseImageRect.width || !baseImageRect.height || !frameRect.width || !frameRect.height) {
      return;
    }

    const next = clampFrameTransform(baseImageRect, frameRect, scale + delta, offset.x, offset.y);
    setScale(next.scale);
    setOffset({ x: next.offsetX, y: next.offsetY });
  }

  function resetFrame() {
    const next = clampFrameTransform(baseImageRect, frameRect, 1, 0, 0);
    setScale(next.scale);
    setOffset({ x: next.offsetX, y: next.offsetY });
  }

  function handleAnalyze() {
    onAnalyze(
      computeCropRectFromFrame(photo, previewSize.width, previewSize.height, scale, offset.x, offset.y)
    );
  }

  const imagePanResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          panStart.current = offset;
        },
        onPanResponderMove: (_, gestureState) => {
          if (!baseImageRect.width || !baseImageRect.height || !frameRect.width || !frameRect.height) {
            return;
          }

          const next = clampFrameTransform(
            baseImageRect,
            frameRect,
            scale,
            panStart.current.x + gestureState.dx,
            panStart.current.y + gestureState.dy
          );
          setOffset({ x: next.offsetX, y: next.offsetY });
        },
      }),
    [baseImageRect, frameRect, offset, scale]
  );

  function handlePreviewLayout(event: LayoutChangeEvent) {
    const { width, height } = event.nativeEvent.layout;
    setPreviewSize({ width, height });
  }

  return (
    <View style={styles.cropScreen}>
      <View style={styles.navRow}>
        <Pressable onPress={onBack}>
          <Text style={styles.backLink}>Back</Text>
        </Pressable>
        <View style={styles.inlinePill}>
          <Text style={styles.inlinePillText}>Position Sample</Text>
        </View>
      </View>

      <View style={[styles.panelCard, styles.cropPanelCard]}>
        <Text style={styles.sectionTitle}>Position the Color</Text>
        <Text style={styles.subtleCopy}>Drag the photo. Zoom until the pink area sits inside the box.</Text>

        <View style={styles.cropPreviewFrame} onLayout={handlePreviewLayout}>
          <View style={styles.cropTouchLayer} {...imagePanResponder.panHandlers}>
            <Image
              source={{ uri: photo.originalUri }}
              style={[
                styles.framingImage,
                {
                  left: transformedImageRect.x,
                  top: transformedImageRect.y,
                  width: transformedImageRect.width,
                  height: transformedImageRect.height,
                },
              ]}
            />
          </View>
          {frameRect.width > 0 && (
            <>
              <View
                pointerEvents="none"
                style={[
                  styles.cropShade,
                  { left: 0, top: 0, width: previewSize.width, height: frameRect.y },
                ]}
              />
              <View
                pointerEvents="none"
                style={[
                  styles.cropShade,
                  {
                    left: 0,
                    top: frameRect.y,
                    width: frameRect.x,
                    height: frameRect.height,
                  },
                ]}
              />
              <View
                pointerEvents="none"
                style={[
                  styles.cropShade,
                  {
                    left: frameEdge.left,
                    top: frameRect.y,
                    width: previewSize.width - frameEdge.left,
                    height: frameRect.height,
                  },
                ]}
              />
              <View
                pointerEvents="none"
                style={[
                  styles.cropShade,
                  {
                    left: 0,
                    top: frameEdge.top,
                    width: previewSize.width,
                    height: previewSize.height - frameEdge.top,
                  },
                ]}
              />
              <View pointerEvents="none" style={[styles.cropOverlay, frameRect]}>
                <View style={styles.cropGridRow}>
                  <View style={styles.cropGridLineVertical} />
                  <View style={styles.cropGridLineVertical} />
                </View>
                <View style={styles.cropGridRow}>
                  <View style={styles.cropGridLineVertical} />
                  <View style={styles.cropGridLineVertical} />
                </View>
              </View>
            </>
          )}
        </View>

        <View style={styles.zoomRow}>
          <Pressable style={styles.zoomButton} onPress={() => adjustZoom(-ZOOM_STEP)}>
            <Text style={styles.zoomButtonText}>-</Text>
          </Pressable>
          <View style={styles.zoomValuePill}>
            <Text style={styles.zoomValueText}>{scale.toFixed(1)}x</Text>
          </View>
          <Pressable style={styles.zoomButton} onPress={() => adjustZoom(ZOOM_STEP)}>
            <Text style={styles.zoomButtonText}>+</Text>
          </Pressable>
        </View>

        <View style={styles.buttonStack}>
          <Pressable style={styles.secondaryButton} onPress={resetFrame}>
            <Text style={styles.secondaryButtonText}>Reset</Text>
          </Pressable>
          <Pressable style={styles.accentButton} onPress={handleAnalyze}>
            <Text style={styles.accentButtonText}>Analyze</Text>
          </Pressable>
        </View>

        {!!errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}
      </View>
    </View>
  );
}

function ProcessingScreen() {
  return (
    <View style={styles.processingScreen}>
      <View style={styles.processingCard}>
        <ActivityIndicator size="large" color="#fff3d5" />
        <Text style={styles.processingTitle}>Analyzing</Text>
        <Text style={styles.processingCopy}>Reading sample color and estimating chlorine concentration.</Text>
      </View>
    </View>
  );
}

function ResultScreen({
  currentScan,
  photo,
  result,
  scanHistory,
  onNewScan,
  onOpenHistory,
  onRetake,
}: {
  currentScan: ScanRecord | null;
  photo: SelectedPhoto | null;
  result: PredictionResult;
  scanHistory: ScanRecord[];
  onNewScan: () => void;
  onOpenHistory: () => void;
  onRetake: () => void;
}) {
  const description = describeReading(result);

  return (
    <ScrollView contentContainerStyle={[styles.screenScroll, styles.screenScrollGrow]}>
      <View style={styles.navRow}>
        <Pressable onPress={onRetake}>
          <Text style={styles.backLink}>Back</Text>
        </Pressable>
        <View
          style={[
            styles.inlinePill,
            description.tone === 'safe' ? styles.safePill : styles.warningPill,
          ]}
        >
          <Text
            style={[
              styles.inlinePillText,
              description.tone === 'safe' ? styles.safePillText : styles.warningPillText,
            ]}
          >
            {description.badge}
          </Text>
        </View>
      </View>

      <View
        style={[
          styles.resultHero,
          description.tone === 'safe' ? styles.resultHeroSafe : styles.resultHeroWarning,
        ]}
      >
        <Text style={styles.eyebrowLight}>Chlorine Concentration</Text>
        <Text style={styles.resultValue}>{result.concentration.toFixed(2)}</Text>
        <Text style={styles.resultUnit}>{result.unit} free chlorine</Text>
        <Text style={styles.resultCopy}>{description.copy}</Text>
      </View>

      <View style={styles.resultSummaryCard}>
        <DetailRow label="Status" value={description.badge} />
        <DetailRow label="Date" value={formatScanDate(result.timestamp)} />
        <DetailRow label="Time" value={formatScanTime(result.timestamp)} />
        <DetailRow label="Location" value={currentScan?.locationLabel || 'Location unavailable'} />
      </View>

      {photo && (
        <View style={styles.resultPreviewCard}>
          <Text style={styles.previewCardTitle}>Analyzed Image</Text>
          <Image
            source={{ uri: photo.uploadUri || photo.originalUri }}
            style={styles.resultPreviewImage}
            resizeMode="cover"
          />
        </View>
      )}

      {scanHistory.length > 0 && (
        <HistoryCard
          title="Saved History"
          subtitle="Your latest readings"
          scans={scanHistory.slice(0, 3)}
        />
      )}

      <Pressable style={styles.secondaryButton} onPress={onOpenHistory}>
        <Text style={styles.secondaryButtonText}>View Previous Scans</Text>
      </Pressable>

      <View style={styles.bottomActions}>
        <Pressable style={styles.primaryButton} onPress={onNewScan}>
          <Text style={styles.primaryButtonText}>New Scan</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
    </View>
  );
}

function HistoryCard({
  title,
  subtitle,
  scans,
}: {
  title: string;
  subtitle: string;
  scans: ScanRecord[];
}) {
  return (
    <View style={styles.resultPreviewCard}>
      <Text style={styles.previewCardTitle}>{title}</Text>
      <Text style={styles.subtleCopy}>{subtitle}</Text>
      <View style={styles.historyList}>
        {scans.map((scan) => (
          <View key={scan.id} style={styles.historyRow}>
            <View style={styles.historyReading}>
              <Text style={styles.historyValue}>
                {scan.concentration.toFixed(2)} {scan.unit}
              </Text>
              <Text style={styles.historyMeta}>
                {formatScanDate(scan.timestamp)} at {formatScanTime(scan.timestamp)}
              </Text>
            </View>
            <Text style={styles.historyLocation}>{scan.locationLabel}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

function HistoryScreen({
  scans,
  onBack,
  onDelete,
}: {
  scans: ScanRecord[];
  onBack: () => void;
  onDelete: (recordId: string) => Promise<void>;
}) {
  function confirmDelete(scan: ScanRecord) {
    Alert.alert(
      'Delete scan?',
      `Remove the ${scan.concentration.toFixed(2)} ${scan.unit} scan from ${formatScanDate(scan.timestamp)}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            void onDelete(scan.id);
          },
        },
      ]
    );
  }

  return (
    <ScrollView contentContainerStyle={[styles.screenScroll, styles.screenScrollGrow]}>
      <View style={styles.navRow}>
        <Pressable onPress={onBack}>
          <Text style={styles.backLink}>Back</Text>
        </Pressable>
      </View>

      <View style={styles.panelCard}>
        <Text style={styles.sectionTitle}>Previous Scans</Text>
        <Text style={styles.subtleCopy}>Review saved chlorine readings and remove any you no longer need.</Text>
      </View>

      {scans.length === 0 ? (
        <View style={styles.resultPreviewCard}>
          <Text style={styles.previewCardTitle}>No scans yet</Text>
          <Text style={styles.subtleCopy}>Your saved readings will appear here after your first completed scan.</Text>
        </View>
      ) : (
        <View style={styles.historyScreenList}>
          {scans.map((scan) => (
            <View key={scan.id} style={styles.historyScreenCard}>
              {(() => {
                const description = describeConcentration(scan.concentration);

                return (
                  <>
                    <View style={styles.historyScreenHeader}>
                      <View style={styles.historyHeaderMain}>
                        <Text style={styles.historyValue}>
                          {scan.concentration.toFixed(2)} {scan.unit}
                        </Text>
                        <View
                          style={[
                            styles.inlinePill,
                            styles.historyBadgeInline,
                            description.tone === 'safe' ? styles.safePill : styles.warningPill,
                          ]}
                        >
                          <Text
                            style={[
                              styles.inlinePillText,
                              description.tone === 'safe' ? styles.safePillText : styles.warningPillText,
                            ]}
                          >
                            {description.badge}
                          </Text>
                        </View>
                      </View>
                      <Pressable style={styles.historyDeleteButton} onPress={() => confirmDelete(scan)}>
                        <Text style={styles.historyDeleteButtonText}>Delete</Text>
                      </Pressable>
                    </View>
                  </>
                );
              })()}
              <Text style={styles.historyMeta}>
                {formatScanDate(scan.timestamp)} at {formatScanTime(scan.timestamp)}
              </Text>
              <Text style={styles.historyLocation}>{scan.locationLabel}</Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}
