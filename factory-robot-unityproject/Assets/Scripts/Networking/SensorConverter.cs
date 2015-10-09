using UnityEngine;
using System.Collections;

// Produces byte arrays that are meant to be sent over the network.
public class SensorConverter : MonoBehaviour {

	// Fragmenting options
	public int nFragments;
	public int imgFragmentLength;

	// Textures
	public RenderTexture cameraRenderTexture;
	private Texture2D readerTexture;
	public int targetTextureSize;
	private int downSampleFactor;

	public RewardManager rewardManager;           // for reward info
	public AgentRemoteControl agentRemoteControl; // for action info

	// Frame syncing
	private byte _frameCounter = 0;
	public byte frameCounter {
		get {
			return _frameCounter;
		}
	}
	private byte frameCounterIncStep = 1;


	void Start() {
		readerTexture = new Texture2D(cameraRenderTexture.width,
		                              cameraRenderTexture.height);
		
		if (cameraRenderTexture.width % targetTextureSize != 0) {
			Debug.LogError("targetTextureSize and cameraRenderTexture.width must be powers of two.");
		}

		downSampleFactor = cameraRenderTexture.width / targetTextureSize;
	}

	public void AdvanceFrameCounter() {
		_frameCounter = (byte) ((_frameCounter + frameCounterIncStep) % 256);
	}

	public byte[][] GetCurrentFragments() {
		byte[][] result = new byte[nFragments][];
		for (int i = 0; i < nFragments; i++) {
			result[i] = GetCurrentFragment(i);
		}
        return result;
	}
	
	private byte[] GetCurrentFragment (int fragmentCounter) {
		Color32[] currentImage = MiscUtils.getCurrentCameraImage(cameraRenderTexture, readerTexture);
		currentImage = MathUtils.downSampleImg(currentImage, downSampleFactor);

		// Protocol (hard coded): 
		//  byte  0: frame counter
		//  byte  1: fragment ID (unsigned)
		//  bytes 2, 3, 4, 5: current total reward (float)
		//  bytes 6..: fragment data
		//  byte -1: checksum
		byte[] buffer = MathUtils.ImageSliceToByteVector(currentImage, 
		                                                 fragmentCounter*imgFragmentLength,
		                                                 (fragmentCounter+1)*imgFragmentLength,
		                                                 7+imgFragmentLength+1,
		                                                 7);
		
		buffer[0] = _frameCounter;
		buffer[1] = (byte)fragmentCounter;

		buffer[2] = (byte)agentRemoteControl.currentAction;

		float[] totalReward = new float[] {rewardManager.GetTotalReward()};
		System.Buffer.BlockCopy(totalReward, 
		                        0, 
		                        buffer, 
		                        3, // offset in sent array
		                        4); 
		
		byte checksum = (byte)MathUtils.GetArraySum(buffer, 0, buffer.Length - 1, 256);
		buffer[buffer.Length - 1] = checksum;
		return buffer;
	}
}
