# External Access Setup Guide for Face Detection System

This guide will help you make your face detection livestream accessible from outside your local network, so that authorized users with the link can view the stream remotely.

## Method 1: Port Forwarding (Recommended)

Port forwarding is the most reliable way to make your stream accessible from the internet.

### Step 1: Configure your router

1. Find your router's IP address (usually `192.168.1.1` or `192.168.0.1`)
2. Log in to your router's admin panel (check your router's manual for credentials)
3. Look for "Port Forwarding" or "Virtual Server" settings
4. Create a new port forwarding rule with these settings:
   - External/Public Port: `5000` (or change this in the code if you need to use a different port)
   - Internal/Private Port: `5000`
   - Internal IP Address: (The IP address of your Raspberry Pi, displayed on the face detection web interface)
   - Protocol: `TCP` or `TCP/UDP`
   - Enable/Save the rule

### Step 2: Test your setup

1. Find your external IP address (displayed on the face detection web interface)
2. From a device outside your network (like a mobile phone using cellular data), try accessing:
   `http://YOUR_EXTERNAL_IP:5000`

### Step 3 (Optional): Set up Dynamic DNS

If your ISP changes your IP address frequently, set up a Dynamic DNS service:

1. Create an account with a DDNS provider (DuckDNS, No-IP, or Dynu)
2. Create a hostname (like `mystream.duckdns.org`)
3. Install the DDNS client on your Raspberry Pi:

```bash
# For DuckDNS
mkdir ~/duckdns
cd ~/duckdns
echo "echo url=\"https://www.duckdns.org/update?domains=YOUR_DOMAIN&token=YOUR_TOKEN&ip=\" | curl -k -o ~/duckdns/duck.log -K -" > duck.sh
chmod 700 duck.sh
crontab -e
# Add this line:
*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1
```

4. Update the `DDNS_HOSTNAME` variable in `face_detection_improved.py`:

```python
DDNS_HOSTNAME = "mystream.duckdns.org"  # Replace with your actual hostname
```

## Method 2: Using LocalTunnel (Fallback)

If port forwarding is blocked by your ISP:

1. Install localtunnel:
```bash
npm install -g localtunnel
```

2. Update the configuration in `face_detection_improved.py`:
```python
EXTERNAL_ACCESS_MODE = 'localtunnel'
```

## Troubleshooting

1. **Stream not accessible externally**:
   - Ensure your router's firewall isn't blocking port 5000
   - Try using a different port (e.g., 8080) and update `PORT` in the code
   - Some ISPs block incoming connections; contact your ISP or try Method 2

2. **Poor performance**:
   - Enable the lower quality stream option
   - Check your upload bandwidth (should be at least 1Mbps for decent quality)

3. **Security concerns**:
   - Consider adding password protection to the stream
   - Only share the access link with trusted individuals
   - For higher security, set up a VPN like WireGuard instead of direct access

## Additional Security Tips

1. Change the default port (5000) to something less common
2. Consider adding HTTP basic authentication
3. Use HTTPS for secure access (requires SSL certificate setup)
4. Limit access to specific IP addresses if possible through your router settings 