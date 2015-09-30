using UnityEngine;
using System.Collections;

public class RewardManager : MonoBehaviour {

	private float totalReward;
//	private float currentReward;

	private float unnormalizedRewardRate;
	public float recentRewardDiscountPerMinute;
	private float decayRate; // per second

	public void PutReward(float reward) {
		totalReward += reward;
		unnormalizedRewardRate += reward;
//		currentReward += reward;
	}

//	public float CollectCurrentReward() {
//		float r = currentReward;
//		currentReward = 0;
//		return r;
//	}

	public float GetTotalReward() {
		return totalReward;
	}

	public float GetRewardRate() {
		return unnormalizedRewardRate * decayRate;
	}

	void Start() {
		decayRate = -Mathf.Log(recentRewardDiscountPerMinute)/60.0f;
	}

	void Update() {
		unnormalizedRewardRate -= unnormalizedRewardRate*Time.smoothDeltaTime*decayRate;
	}
}
