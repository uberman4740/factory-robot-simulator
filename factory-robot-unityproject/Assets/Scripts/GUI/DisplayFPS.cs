using UnityEngine;
using System.Collections;

public class DisplayFPS : MonoBehaviour {
	public Vector2 position;

    void OnGUI() {
        GUI.Label(new Rect(position.x, position.y, 200, 50), 
			          "FPS: " + (1.0f/Time.smoothDeltaTime));

    }
}
