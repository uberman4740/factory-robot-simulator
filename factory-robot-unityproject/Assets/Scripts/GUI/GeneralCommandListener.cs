using UnityEngine;
using System.Collections;

public class GeneralCommandListener : MonoBehaviour {

	public KeyCode exitKey = KeyCode.Escape;
		
	void Update () {
		if (Input.GetKeyDown (exitKey)) {
			Application.Quit();
		}
	}
}
