module.exports = {
  apps: [
    {
      name: 'axsy-inference',
      cwd: '/home/ec2-user/axsy_inference',
      script: 'uvicorn',
      args: ['server:get_app', '--factory', '--host', '0.0.0.0', '--port', '3000', '--workers', '1', '--log-level', 'info'],
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        NODE_ENV: 'production',
        GOOGLE_APPLICATION_CREDENTIALS: '/home/ec2-user/axsy_inference/smart-vision-trainiing-sa.json'
      }
    }
  ]
};



