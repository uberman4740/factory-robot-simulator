using UnityEngine;
using System.Collections;

public class DisplayFPS : MonoBehaviour {
	public Vector2 position;
	public TimeStepManager timeStepManager;

	private int timeStepCounter = 0;
	private float lastTimeStep;

	private float currentFpsSmoothing = 0.9f;
	private float currentFps = 0.0f;
	private float averageFps;
		
	private float firstStepTime = -1.0f;

	void Start() {
		lastTimeStep = Time.realtimeSinceStartup;
	}

	void Update() {

		if (timeStepManager.state == TimeStepManager.State.Advance) {

			timeStepCounter++;

			float newFps = 1.0f / (Time.realtimeSinceStartup - lastTimeStep);
			lastTimeStep = Time.realtimeSinceStartup;

			currentFps = currentFpsSmoothing * currentFps + (1 - currentFpsSmoothing) * newFps;

			if (timeStepCounter == 2) {
				firstStepTime = Time.realtimeSinceStartup;	
			}
			if (timeStepCounter > 2) {
				averageFps = (timeStepCounter-2) / (Time.realtimeSinceStartup - firstStepTime);
			}

		}
	}

    void OnGUI() {
        GUI.Label(new Rect(position.x, position.y, 200, 50), 
			          "FPS: " + (1.0f/Time.smoothDeltaTime));

		GUI.Label(new Rect(position.x, position.y + 20, 500, 50), 
		          "SPS (curr.: " + currentFps + ")");

		GUI.Label(new Rect(position.x, position.y + 40, 200, 50), 
		          "SPS (total: " + averageFps + ")");

    }
}
