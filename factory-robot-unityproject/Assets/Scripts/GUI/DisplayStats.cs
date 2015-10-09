using UnityEngine;
using System.Collections;

public class DisplayStats : MonoBehaviour {
	public RewardManager rewardManager;

	public InputListener inputListener;

	public Vector2 position;


	void OnGUI() {
		GUI.Label(new Rect(position.x, position.y, 400, 50), 
		          "Reward: " + rewardManager.GetTotalReward());
		GUI.Label(new Rect(position.x, position.y + 20, 400, 50), 
		          "Current reward rate: " + rewardManager.GetRewardRate() + " / sec");
		GUI.Label(new Rect(position.x, position.y + 40, 400, 50), 
		          "Current input action: " + inputListener.currentInputAction);

	}
}
