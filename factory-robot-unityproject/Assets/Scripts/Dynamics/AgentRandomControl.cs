using UnityEngine;
using System.Collections;

public class AgentRandomControl : MonoBehaviour {

	public float drivingSpeed;
	public float turningSpeed;
	public float deltaTime;
	public int actionDurationLower;
	public int actionDurationUpper;

	public int currentAction;

	private int nextActionSwitchFrame;
	private int totalFrames;

	void Start() {
		nextActionSwitchFrame = 0;
	}

	void Update () {

		if (totalFrames >= nextActionSwitchFrame) {
			nextActionSwitchFrame += Random.Range(actionDurationLower, 
			                                      actionDurationUpper);

			currentAction = Random.Range(0, 3);
		}

		switch (currentAction) {
		case 0:
			transform.Rotate(0,-turningSpeed * deltaTime, 0);
			break;
		case 2:
			transform.Rotate(0, turningSpeed * deltaTime, 0);
			break;
		} 

		transform.Translate(Vector3.forward * drivingSpeed * deltaTime);

		totalFrames++;
	}
}
