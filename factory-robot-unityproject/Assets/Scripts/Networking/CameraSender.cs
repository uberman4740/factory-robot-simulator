using UnityEngine;
using System.Collections;
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

	// Use this for initialization
	void Start () {
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        socket.EnableBroadcast = true;

        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);

        if (cameraRenderTexture.width % targetTextureSize != 0) {
            Debug.LogError("targetTextureSize and cameraRenderTexture.width must be powers of two.");
        }
	}
	
	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(KeyCode.A)) {

            //float sum = 0.0f;
            //var floats = new float[4096 * 3];
            //for (int i = 0; i < floats.Length; i++) {
            //    floats[i] = Random.Range(0.0f, 1.0f);
            //    sum += floats[i];
            //}

            //SendFloatArray(floats);

            Color32[] currentImage = MiscUtils.getCurrentCameraImage(cameraRenderTexture, readerTexture);

            int downsampleFactor = cameraRenderTexture.width / targetTextureSize;

            currentImage = MathUtils.downSampleImg(currentImage, downsampleFactor);

            byte[] pixels = MathUtils.ImageToByteVector(currentImage);
            SendByteArray(pixels, preamble);


            //float[] pixels = MathUtils.ImageToFloatVector(currentImage);
            //SendFloatArray(pixels, preamble);
        }
	}

    private void SendFloatArray(float[] floats, byte[] preamble) {
        byte[] bu = new byte[4 * floats.Length];
        System.Buffer.BlockCopy(floats, 0, bu, 0, bu.Length);
        SendByteArray(bu, preamble);
    }

    private void SendByteArray(byte[] bu, byte[] preamble) {
        int nFragments = (bu.Length - 1) / packetLength + 1;
        byte[,] fragments = new byte[nFragments, packetLength];

        System.Buffer.BlockCopy(bu, 0, fragments, 0, bu.Length);

        socket.SendTo(preamble, new IPEndPoint(IPAddress.Broadcast, port));

        for (int i = 0; i < fragments.GetLength(0); i++) {
            byte[] buffer = new byte[packetLength];
            for (int j = 0; j < packetLength; j++) {
                buffer[j] = fragments[i, j];
            }
            socket.SendTo(buffer, new IPEndPoint(IPAddress.Broadcast, port));
        }
        Debug.LogFormat("number of fragments sent: {0}", fragments.GetLength(0));
    }

    void OnApplicationQuit() {
        if (socket != null) {
            //socket.Shutdown(SocketShutdown.Both);
            socket.Close();
        }
    }
}
