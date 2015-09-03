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
}
