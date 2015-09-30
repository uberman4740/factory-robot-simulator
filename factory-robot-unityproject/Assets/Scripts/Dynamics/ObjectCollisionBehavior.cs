using UnityEngine;
using System.Collections;

public class ObjectCollisionBehavior : MonoBehaviour {
	private RewardManager rewardManager;

	private AgentRemoteControl agentRemoteControl;

	void Awake() {
		rewardManager = transform.GetComponentInParent<RewardManager>();
		agentRemoteControl = transform.GetComponentInParent<AgentRemoteControl>();
	}

	void OnTriggerEnter(Collider other) {
		var otherCollisionInfo = other.gameObject.GetComponent<CollisionInfo>();
		if (otherCollisionInfo == null) return;
		otherCollisionInfo.Notify(transform.forward);

		// Apply reward only if agent is moving forward.
		if (agentRemoteControl != null &&
		    agentRemoteControl.GetCurrentSpeed() <= 0) {
			return;
		}

		rewardManager.PutReward(
			otherCollisionInfo.GetCollisionReward());
	}
}
