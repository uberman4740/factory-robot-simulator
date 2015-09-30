using UnityEngine;
using System.Collections;

public class AgentRemoteControl : MonoBehaviour {

	public InputListener inputListener;

	public float speed;
	public float turningSpeed;

	public float acceleration;
	public float deceleration;

	public float revSpeed;

	private float currentSpeed;
	

	void Update () {
		bool accelerate = true;
		int action = inputListener.GetCurrentInput();

		if (Input.GetKey(KeyCode.DownArrow)) {
			action = 3;
		}

		switch (action) {
		case 0:
			transform.Rotate(0,-turningSpeed*Time.smoothDeltaTime, 0);
			break;
		case 2:
			transform.Rotate(0, turningSpeed*Time.smoothDeltaTime, 0);
			break;
		case 3:
			accelerate = false;
			break;
		} 

		if (accelerate) {
			if (currentSpeed < speed) {
				currentSpeed += acceleration * Time.smoothDeltaTime;
				if (currentSpeed > speed) {
					currentSpeed = speed;
				}
			}
		} else {
			if (currentSpeed > revSpeed) {
				currentSpeed -= deceleration * Time.smoothDeltaTime;
				if (currentSpeed < revSpeed) {
					currentSpeed = revSpeed;
				}
			}
		}

		transform.Translate(Vector3.forward * currentSpeed * Time.smoothDeltaTime);
	}

	public float GetCurrentSpeed() {
		return currentSpeed;
	}
}
