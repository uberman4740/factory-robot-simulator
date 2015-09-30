using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class CameraSwitcher : MonoBehaviour {

	public List<Camera> cameras;
	public KeyCode switchKey;

	private int currentCamera = 0;

	void Start() {
		if (cameras.Count == 0) {
			Debug.LogError("No cameras specified for CameraSwitcher.");
		}
		cameras[0].gameObject.SetActive(true);
		for (int i = 1; i < cameras.Count; i++) {
			cameras[i].gameObject.SetActive(false);
		}
	}

	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(switchKey)) {
//			if (currentCamera > -1) {
//				cameras[currentCamera].gameObject.SetActive(false);
//			}
//			if (currentCamera == cameras.Count-1) {
//				currentCamera = -1;
//			} else {
//				currentCamera++;
//				cameras[currentCamera].gameObject.SetActive(true);
//			}
			cameras[currentCamera].gameObject.SetActive(false);
			currentCamera = (currentCamera + 1) % cameras.Count;
			cameras[currentCamera].gameObject.SetActive(true);
        }
	}
}
