#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import rospy
import actionlib
import signal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry

robot_position = {'x': 0.0, 'y': 0.0}
position_lock = threading.Lock()

def update_position(odom_msg):
    with position_lock:
        robot_position['x'] = odom_msg.pose.pose.position.x
        robot_position['y'] = odom_msg.pose.pose.position.y

class AnomalyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_content = '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Anomaly Detection Robot Control</title>
                <style>
                    body {
                        background-color: #1C2526;
                        color: #E0E7E9;
                        font-family: 'Roboto', sans-serif;
                        margin: 0;
                        padding: 20px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        min-height: 100vh;
                    }
                    header {
                        background-color: #0A0E14;
                        width: 100%;
                        padding: 20px;
                        text-align: center;
                        font-size: 2em;
                        color: #00A7E1;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
                    }
                    main {
                        text-align: center;
                        padding: 20px;
                        width: 100%;
                        max-width: 800px;
                    }
                    button {
                        background-color: #D90429;
                        color: #E0E7E9;
                        border: none;
                        padding: 20px 40px;
                        font-size: 1.5em;
                        cursor: pointer;
                        border-radius: 10px;
                        transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
                        box-shadow: 0 0 15px rgba(217, 4, 41, 0.5);
                        margin: 20px 0;
                    }
                    button:hover {
                        background-color: #EF233C;
                        transform: scale(1.05);
                        box-shadow: 0 0 25px rgba(217, 4, 41, 0.7);
                    }
                    button:active {
                        transform: scale(1);
                        background-color: #8D0022;
                    }
                    .status {
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #2D3B45;
                        border-radius: 8px;
                        font-size: 1.2em;
                        color: #A3BFFA;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                    }
                    .video-container {
                        margin: 20px 0;
                        padding: 10px;
                        background-color: #0A0E14;
                        border-radius: 8px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
                    }
                    iframe {
                        width: 100%;
                        height: 400px;
                        border: none;
                        border-radius: 8px;
                    }
                    .message {
                        margin: 15px 0;
                        padding: 10px;
                        background-color: #00A7E1;
                        color: #E0E7E9;
                        border-radius: 8px;
                        font-size: 1em;
                        display: none;
                    }
                </style>
                <script>
                    function updatePosition() {
                        fetch('/position')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('robot-x').textContent = data.x.toFixed(2);
                                document.getElementById('robot-y').textContent = data.y.toFixed(2);
                            })
                            .catch(error => console.error('Error getting position:', error));
                    }

                    function sendAnomalyResponse() {
                        fetch('/move', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ x: 0.991, y: -0.003 })
                        })
                        .then(response => response.json())
                        .then(data => {
                            showMessage(data.message);
                        })
                        .catch(error => console.error('Error:', error));
                    }

                    function showMessage(message) {
                        const messageDiv = document.getElementById('message');
                        messageDiv.textContent = message;
                        messageDiv.style.display = 'block';
                        setTimeout(() => {
                            messageDiv.style.display = 'none';
                        }, 3000);
                    }

                    window.onload = function() {
                        updatePosition();
                        setInterval(updatePosition, 1000);
                    };
                </script>
            </head>
            <body>
                <header>
                    Anomaly Detection Robot Control
                </header>
                <main>
                    <button onclick="sendAnomalyResponse()">Respond to Anomaly</button>
                    <div class="status">
                        Robot Position: x: <span id="robot-x">0.00</span>, y: <span id="robot-y">0.00</span>
                    </div>
                    <div class="video-container">
                        <iframe src="http://192.168.149.1:9000/" frameborder="0"></iframe>
                    </div>
                    <div id="message" class="message"></div>
                </main>
            </body>
            </html>
            '''
            self.wfile.write(html_content.encode())
        elif self.path == '/position':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with position_lock:
                self.wfile.write(json.dumps(robot_position).encode())

    def do_POST(self):
        if self.path == '/move':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            threading.Thread(target=self.move_robot, args=(x, y)).start()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'message': 'Navigating to anomaly'}).encode())

    def move_robot(self, x, y):
        print(f"Moving robot to ({x}, {y})")
        if not rospy.core.is_initialized():
            rospy.init_node('anomaly_move', anonymous=True)
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x, y, 0.0)
        goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        client.send_goal(goal)
        client.wait_for_result()
        with position_lock:
            robot_position['x'] = x
            robot_position['y'] = y

def run(server_class=HTTPServer, handler_class=AnomalyHandler, port=8000):
    rospy.init_node('anomaly_move', anonymous=True)
    rospy.Subscriber('/odom', Odometry, update_position)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    def signal_handler(sig, frame):
        print('Shutting down server...')
        httpd.server_close()
        print('Server stopped')
    signal.signal(signal.SIGINT, signal_handler)
    httpd.serve_forever()

if __name__ == '__main__':
    run()
