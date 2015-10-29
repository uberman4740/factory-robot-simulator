using UnityEngine;
using System.Collections;

public class SequentialData : MonoBehaviour {
	public bool storeImages = true;

	public RenderTexture cameraRenderTexture;
	private Texture2D readerTexture;
	public Camera agentCamera;

	public string[] recordCategoryTags;

	public string trainingFilePath;
	public string labelFileName = "labels.dat";
	public string captureFilePrefix = "capture";
	public string actionFileName = "actions.dat";

	public int nDirectionSensors = 7;
	
	public string wallTag = "Building";

	public AgentRandomControl agentRandomControl;
//	public PeriodicResetter periodicResetter;
	
	int counter = 0;

	void Start () {
		readerTexture = new Texture2D(cameraRenderTexture.width,
		                              cameraRenderTexture.height);

		FileUtils.CopyFile(trainingFilePath + labelFileName, trainingFilePath + labelFileName + "_backup");
		FileUtils.WriteStringToFile(trainingFilePath + labelFileName, "");

		FileUtils.CopyFile(trainingFilePath + actionFileName, trainingFilePath + actionFileName + "_backup");
		FileUtils.WriteStringToFile(trainingFilePath + actionFileName, "");
	}
	
	void Update () {
		LabeledDataUtil.Snapshot(agentCamera,
		                        cameraRenderTexture,
		                        readerTexture,
		                        trainingFilePath,
		                        captureFilePrefix,
		                        labelFileName,
		                        recordCategoryTags,
		                        nDirectionSensors,
		                        wallTag,
		                        counter,
		                        wallHitMultiplier: 0.4f,
		                        storeImages: storeImages);	

		if (agentRandomControl != null) {
			FileUtils.AppendStringToFile(trainingFilePath + actionFileName, agentRandomControl.currentAction.ToString() + '\n');
		}

		counter++;
		if (counter % 1000 == 0) {
			Debug.LogFormat("counter: {0}", counter);
		}
//		if (periodicResetter.HasBeenReset()) {
//			Debug.Log ("Reset!");
//		}
	}
}
