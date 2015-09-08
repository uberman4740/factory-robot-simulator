using UnityEngine;
using System.Collections;

public class CameraSwitcher : MonoBehaviour {

	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(KeyCode.C)) {
            Camera.main.depth = 1.0f - Camera.main.depth;
        }

        if (Input.GetKeyDown(KeyCode.Escape)) {
            Application.Quit();
        }
	}
}
