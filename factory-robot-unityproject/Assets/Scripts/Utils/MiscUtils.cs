using UnityEngine;
using System.Collections;

public static class MiscUtils {

    public static Color32[] getCurrentCameraImage(RenderTexture cameraRenderTexture, Texture2D readerTexture) {
        RenderTexture.active = cameraRenderTexture;
        readerTexture.ReadPixels(new Rect(0, 0,
                                          cameraRenderTexture.width,
                                          cameraRenderTexture.height),
                                 0, 0);
        readerTexture.Apply();
        return readerTexture.GetPixels32();
    }

    public static int IndexOf(object[] arr, object comp) {
        for (int i = 0; i < arr.Length; i++) {
            if (arr[i].Equals(comp)) {
                return i;
            }
        }
        return -1;
    }
}
