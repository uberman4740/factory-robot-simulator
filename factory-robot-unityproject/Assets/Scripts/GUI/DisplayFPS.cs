using UnityEngine;
using System.Collections;

public class DisplayFPS : MonoBehaviour {
    void OnGUI() {
        GUI.Label(new Rect(10, 10, 200, 50), 
			          "FPS: " + (1.0f/Time.smoothDeltaTime));

    }
}
