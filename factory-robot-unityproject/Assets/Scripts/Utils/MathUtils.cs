using UnityEngine;
using System.Collections;

public static class MathUtils {
    public static Vector3 getUniformRandomVector3(Vector3 center, Vector3 range) {
        return new Vector3(Random.Range(center.x - range.x / 2, center.x + range.x / 2),
                           Random.Range(center.y - range.y / 2, center.y + range.y / 2),
                           Random.Range(center.z - range.z / 2, center.z + range.z / 2));
    }



    public static float[] downSampleImg(float[] orig, int factor) {
        int l = Mathf.RoundToInt(Mathf.Sqrt(orig.Length));
        if (l * l != orig.Length) {
            Debug.LogError("Image not a square!");
            return null;
        }
        if (l % factor != 0) {
            Debug.LogError("Scale factor must divide Image length!");
            return null;
        }

        if (factor == 1) {
            return orig;
        }

        int newL = l / factor;
        float[] result = new float[newL * newL];
        for (int y = 0; y < newL; y++) {
            for (int x = 0; x < newL; x++) {
                float total = 0.0f;
                for (int i = 0; i < factor; i++) {
                    for (int j = 0; j < factor; j++) {
                        total += orig[l * (factor * y + i) + factor * x + j];
                    }
                }
                result[newL * y + x] = total / (factor * factor);
            }
        }
        return result;

    }

    public static Color32[] downSampleImg(Color32[] orig, int factor) {
        int l = Mathf.RoundToInt(Mathf.Sqrt(orig.Length));
        if (l * l != orig.Length) {
            Debug.LogError("Image not a square!");
            return null;
        }
        if (l % factor != 0) {
            Debug.LogError("Scale factor must divide Image length!");
            return null;
        }
        int newL = l / factor;
        Color32[] result = new Color32[newL * newL];
        for (int y = 0; y < newL; y++) {
            for (int x = 0; x < newL; x++) {
                float totalR = 0.0f;
                float totalG = 0.0f;
                float totalB = 0.0f;

                for (int i = 0; i < factor; i++) {
                    for (int j = 0; j < factor; j++) {
                        totalR += orig[l * (factor * y + i) + factor * x + j].r;
                        totalG += orig[l * (factor * y + i) + factor * x + j].g;
                        totalB += orig[l * (factor * y + i) + factor * x + j].b;
                    }
                }
                byte valueR = (byte)Mathf.RoundToInt(totalR / (factor * factor));
                byte valueG = (byte)Mathf.RoundToInt(totalG / (factor * factor));
                byte valueB = (byte)Mathf.RoundToInt(totalB / (factor * factor));
                result[newL * y + x] = new Color32(valueR, valueG, valueB, 255);
            }
        }
        return result;
    }

    public static float[] ImageToFloatVector(Color32[] orig, bool flipY = true) {
        int l = Mathf.RoundToInt(Mathf.Sqrt(orig.Length));
        if (l * l != orig.Length) {
            Debug.LogError("Image not a square!");
            return null;
        }

        float[] result = new float[3*orig.Length];
        for (int i = 0; i < orig.Length; i++) {
            int tI = flipY ? l*(l - 1 - (i/l)) + i % l : i;

            result[3 * i] = orig[tI].r / 255.0f;
            result[3 * i + 1] = orig[tI].g / 255.0f;
            result[3 * i + 2] = orig[tI].b / 255.0f;
        }
        return result;
    }

    public static byte[] ImageToByteVector(Color32[] orig, bool flipY = true) {
        int l = Mathf.RoundToInt(Mathf.Sqrt(orig.Length));
        if (l * l != orig.Length) {
            Debug.LogError("Image not a square!");
            return null;
        }

        byte[] result = new byte[3 * orig.Length];
        for (int i = 0; i < orig.Length; i++) {
            int tI = flipY ? l * (l - 1 - (i / l)) + i % l : i;

            result[3 * i] = orig[tI].r;
            result[3 * i + 1] = orig[tI].g;
            result[3 * i + 2] = orig[tI].b;
        }
        return result;
    }
}
