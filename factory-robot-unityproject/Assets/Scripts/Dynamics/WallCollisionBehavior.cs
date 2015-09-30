using UnityEngine;
using System.Collections;

public class WallCollisionBehavior : MonoBehaviour {
	private AgentController agentController;

	public float wallHitReward = -10.0f;
	public RewardManager rewardManager;

	void Awake() {
		agentController = transform.root.GetComponent<AgentController>();
	}

	void OnTriggerEnter(Collider other) {
		if (!other.gameObject.CompareTag ("Building")) {
			return;
		}

		rewardManager.PutReward(wallHitReward);

		TurnTowardCenter();

	}


	void TurnTowardCenter() {
		Vector3 centerDirection = -transform.position;
		float rot = Mathf.Atan2(centerDirection.x, centerDirection.z);

		var newRotation = new Quaternion();
		newRotation.SetLookRotation(centerDirection, Vector3.up);
		transform.root.localRotation = newRotation;
		transform.root.Translate(0.0f, 0.0f, 1.0f);

		agentController.SetCurrentAngle(rot * Mathf.Rad2Deg);
	}
}
