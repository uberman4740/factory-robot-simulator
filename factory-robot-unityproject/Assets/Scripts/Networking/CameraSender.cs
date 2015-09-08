using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;

public class CameraSender : MonoBehaviour {

    public int port = 8888;
    public RenderTexture cameraRenderTexture;
    private Texture2D readerTexture;

    public int targetTextureSize = 64;

    private int packetLength = 1024;

    private Socket socket;
    private byte[] preamble = new byte[] { 1, 2, 3, 4, 5, 6 };

    public float frameSendPeriod = 0.5f;
    public float fragmentSendPeriod = 0.3f / 12;

    private float nextFrameTime;
    private float nextFragmentTime;

    private List<byte[]> remainingFragments = new List<byte[]>();

    private IPEndPoint targetDestination;

	// Use this for initialization
	void Start () {
        targetDestination = new IPEndPoint(IPAddress.Broadcast, port);

        socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        socket.EnableBroadcast = true;

        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);

        if (cameraRenderTexture.width % targetTextureSize != 0) {
            Debug.LogError("targetTextureSize and cameraRenderTexture.width must be powers of two.");
        }

        nextFrameTime = Time.time + frameSendPeriod;
        nextFragmentTime = Time.time + fragmentSendPeriod;
	}
	
	// Update is called once per frame
	void Update () {
        if (Time.time > nextFrameTime) {
            if (remainingFragments.Count > 0) {
                Debug.LogWarning("Fragments not being sent frequently enough!");
                remainingFragments.Clear();
            }


            Color32[] currentImage = MiscUtils.getCurrentCameraImage(cameraRenderTexture, readerTexture);

            int downsampleFactor = cameraRenderTexture.width / targetTextureSize;

            currentImage = MathUtils.downSampleImg(currentImage, downsampleFactor);

            byte[] pixels = MathUtils.ImageToByteVector(currentImage);
            ScheduleSendByteArray(pixels, preamble);


            //float[] pixels = MathUtils.ImageToFloatVector(currentImage);
            //SendFloatArray(pixels, preamble);

            nextFrameTime = Time.time + frameSendPeriod;
        }

        if (Time.time > nextFragmentTime) {
            if (remainingFragments.Count > 0) {
                SendBuffer(remainingFragments[0], targetDestination);
                remainingFragments.RemoveAt(0);
            }
            nextFragmentTime = Time.time + fragmentSendPeriod;
        }

	}

    private void ScheduleSendFloatArray(float[] floats, byte[] preamble) {
        byte[] bu = new byte[4 * floats.Length];
        System.Buffer.BlockCopy(floats, 0, bu, 0, bu.Length);
        ScheduleSendByteArray(bu, preamble);
    }

    private void ScheduleSendByteArray(byte[] bu, byte[] preamble) {
        int nFragments = (bu.Length - 1) / packetLength + 1;
        byte[,] fragments = new byte[nFragments, packetLength];

        System.Buffer.BlockCopy(bu, 0, fragments, 0, bu.Length);

        //socket.SendTo(preamble, new IPEndPoint(IPAddress.Broadcast, port));
        remainingFragments.Add(preamble);

        for (int i = 0; i < fragments.GetLength(0); i++) {
            byte[] buffer = new byte[packetLength];
            for (int j = 0; j < packetLength; j++) {
                buffer[j] = fragments[i, j];
            }
            remainingFragments.Add(buffer);    
        }
        //Debug.LogFormat("number of fragments sent: {0}", fragments.GetLength(0));
    }

    private void SendBuffer(byte[] buffer, IPEndPoint destination) {
        socket.SendTo(buffer, destination);
    }

    void OnApplicationQuit() {
        if (socket != null) {
            //socket.Shutdown(SocketShutdown.Both);
            socket.Close();
        }
    }
}
