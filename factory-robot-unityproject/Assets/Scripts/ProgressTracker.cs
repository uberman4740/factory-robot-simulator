using UnityEngine;
using System.Collections;

public class ProgressTracker : MonoBehaviour {

	public RewardManager rewardManager;
	public string progressFileName;	
	public float recordingPeriod;
	private float nextRecordingTime;

	void Start () {
		nextRecordingTime = Time.time + recordingPeriod;
		FileUtils.CopyFile(progressFileName, progressFileName + "_bak");
	}
	
	// Update is called once per frame
	void Update () {
		if (Time.time > nextRecordingTime) {
			nextRecordingTime += recordingPeriod;
			FileUtils.AppendStringToFile(progressFileName, 
			                             Time.time + "," + rewardManager.GetRewardRate() + "\n");
		}
	}		
}
