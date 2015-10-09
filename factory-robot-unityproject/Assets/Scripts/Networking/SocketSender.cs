using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;


public class SocketSender : MonoBehaviour {
	public string targetIp;
	public int port = 8888;
	
	private Socket socket;
	private IPEndPoint targetDestination;

	void Start () {
		targetDestination = new IPEndPoint(IPAddress.Parse(targetIp), port);
		
		socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
		socket.EnableBroadcast = true;		
	}

    /** Sends the given sequence of frames to this sender's target */
    public void SendFrame(byte[][] frame) {
        for (int i = 0; i < frame.Length; i++) {
            SendBuffer(frame[i], targetDestination);
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
