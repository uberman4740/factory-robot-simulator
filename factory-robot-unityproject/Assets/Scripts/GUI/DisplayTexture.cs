using UnityEngine;
using System.Collections;

public class DisplayTexture : MonoBehaviour {
    public int x = 0;
    public int y = 0;
    public int w = 256;
    public int h = 256;

    public Texture texture;

    void OnGUI() {
        Graphics.DrawTexture(new Rect(x, y, w, h), texture);
    }
}
