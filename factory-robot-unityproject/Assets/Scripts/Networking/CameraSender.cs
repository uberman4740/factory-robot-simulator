using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;

public class CameraSender : MonoBehaviour {

    public string targetIp = "192.168.178.20";
    public int port = 8888;

    private Socket socket;
    private IPEndPoint targetDestination;

    public RenderTexture cameraRenderTexture;
    private Texture2D readerTexture;
    public int targetTextureSize = 64;
    
    public int packetLength = 1024;
    public int nFragments = 12;
    public float fragmentSendPeriod = 0.1f / 12;


    private float nextFragmentTime;
    private int fragmentCounter = 0;


	// Use this for initialization
	void Start () {
        targetDestination = new IPEndPoint(IPAddress.Parse(targetIp), port);

        socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        socket.EnableBroadcast = true;

        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);

        if (cameraRenderTexture.width % targetTextureSize != 0) {
            Debug.LogError("targetTextureSize and cameraRenderTexture.width must be powers of two.");
        }
        nextFragmentTime = Time.time + fragmentSendPeriod;
	}
	
	// Update is called once per frame
	void Update () {
        if (Time.time > nextFragmentTime) {
            nextFragmentTime = Time.time + fragmentSendPeriod;

            Color32[] currentImage = MiscUtils.getCurrentCameraImage(cameraRenderTexture, readerTexture);

            byte[] pixels = MathUtils.ImageSliceToByteVector(currentImage, 
                fragmentCounter*packetLength,
                (fragmentCounter+1)*packetLength,
                1);
            
            pixels[0] = (byte)fragmentCounter;

            SendBuffer(pixels, targetDestination);
            fragmentCounter = (fragmentCounter + 1) % nFragments;
        }
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
