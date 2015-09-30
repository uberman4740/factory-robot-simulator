using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;

public class InputListener : MonoBehaviour {

	public int port = 8889;
	private IPAddress ip = IPAddress.Any;

	private Socket socket;
	private byte[] buffer = new byte[16];

	private int currentInput;


	void Start () {
		socket = new Socket(AddressFamily.InterNetwork, 
		                    SocketType.Dgram, 
		                    ProtocolType.Udp);

		socket.Bind(new IPEndPoint(ip, port));
		socket.Blocking = false;

		StartCoroutine(Poll());
	}

	IEnumerator Poll() {
		while (true) {
			yield return null;
			if (socket.Poll(0, SelectMode.SelectRead)) {
				int bytesReceived = socket.Receive(buffer, 
				                                   0, 
				                                   buffer.Length, 
				                                   SocketFlags.None);

				if (bytesReceived > 0) {
					currentInput = buffer[0];
				}
			}
		}
	}

	public int GetCurrentInput() {
		return currentInput;
	}

	void OnApplicationQuit() {
		socket.Close();
	}


}
