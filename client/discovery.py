import requests
import socket
import json
import time
import logging

from client.utils.config import DISCOVERY_RETRIES, DISCOVERY_TIMEOUT,SERVICE_NAME

# Standard Python pattern
logger = logging.getLogger(__name__)

def discover_server(timeout=DISCOVERY_TIMEOUT,retries=DISCOVERY_RETRIES):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    sock.bind(('', 0))  
    sock.settimeout(timeout)

    request = {"action": "discover", "service": SERVICE_NAME}
    discovery_ports = [5001, 5002, 5003, 5004, 5005]
    
    for attempt in range(retries):
        logger.info(f"🔍 Discovery attempt {attempt + 1}/{retries}")
        
        # Try all discovery ports in each attempt
        for port in discovery_ports:
            try:
                logger.debug(f"   Broadcasting to port {port}...")
                sock.sendto(json.dumps(request).encode(), ('<broadcast>', port))
                # Zusatz: gerichteter Broadcast auf dein Subnetz (falls globales Broadcast gefiltert wird)
                try: sock.sendto(json.dumps(request).encode(), ('192.168.199.255', port))
                except Exception: pass

                # Wait for response with short timeout per port
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode())
                    
                    if response.get("service") == SERVICE_NAME:
                        SERVER_URL = f"http://{response['ip']}:{response['port']}"
                        logger.info(f"✅ Discovered server at {SERVER_URL} (via port {port})")
                        sock.close()
                        return SERVER_URL
                        
                except socket.timeout:
                    # No response on this port, try next port
                    continue
                    
            except Exception as e:
                logger.debug(f"   Error on port {port}: {e}")
                return "http://10.0.0.1:5000"
        try:
            sock.sendto(json.dumps(request).encode(), ('192.168.196.119', 5001))
        except Exception:
            pass
        
        if attempt < retries - 1:
            logger.warning(f"⚠️  No response on any port, retrying in 1 second...")
            time.sleep(1)
    
    logger.error("❌ Could not discover server after all retries")
    sock.close()
    return None

def heartbeat(SERVER_URL: str):
    """Check if server is running"""
    logger.info(f"\n🔍 Checking server health...")
    
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ Server is healthy!")
            return True
        else: 
            logger.info(f"⚠️  Server responded with {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Cannot connect to server!")
        return False
        
    except requests.exceptions. Timeout:
        logger.error(f"❌ Server timeout!")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error:  {e}")
        return False