using UnityEngine;
using System.Collections;

public class AgentRemoteControl : MonoBehaviour {

	public InputListener inputListener;
	public TimeStepManager timeStepManager;

	/* Dynamics */
	public float speed;
	public float turningSpeed;

	public float acceleration;
	public float deceleration;

	public float revSpeed;
	private float currentSpeed;	

	private int _currentAction;
	public int currentAction {
		get {
			return _currentAction;
		}
	}

	void Update () {
		// Only execute this Update method if TimeStepManager allows it.
		if (timeStepManager.state != TimeStepManager.State.Advance) {
			return;
		}

		float deltaTime = timeStepManager.deltaTime;

		bool accelerate = true;
		_currentAction = inputListener.currentInputAction;


		switch (_currentAction) {
		case 0:
			transform.Rotate(0,-turningSpeed * deltaTime, 0);
			break;
		case 2:
			transform.Rotate(0, turningSpeed * deltaTime, 0);
			break;
		case 3:
			accelerate = false;
			break;
		} 

		if (accelerate) {
			if (currentSpeed < speed) {
				currentSpeed += acceleration * deltaTime;
				if (currentSpeed > speed) {
					currentSpeed = speed;
				}
			}
		} else {
			if (currentSpeed > revSpeed) {
				currentSpeed -= deceleration * deltaTime;
				if (currentSpeed < revSpeed) {
					currentSpeed = revSpeed;
				}
			}
		}

		transform.Translate(Vector3.forward * currentSpeed * deltaTime);
	}
	
	public float GetCurrentSpeed() {
		return currentSpeed;
	}
}
