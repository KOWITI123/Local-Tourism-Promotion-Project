// onesignal-init.js
window.OneSignal = window.OneSignal || [];

// Load the OneSignal SDK
const script = document.createElement('script');
script.src = 'https://cdn.onesignal.com/sdks/OneSignalSDK.js';
script.async = true;
document.head.appendChild(script);

// Initialize OneSignal once the SDK is loaded
script.onload = function() {
  OneSignal.push(function() {
    OneSignal.init({
      appId: "f33799af-cb9e-460e-adac-e4848cfd848b", // Replace with your OneSignal App ID
      safari_web_id: "your-safari-web-id", // Optional, for Safari support
      notifyButton: {
        enable: true, // Show a bell icon for users to manage notifications
      },
      allowLocalhostAsSecureOrigin: true, // For local testing
    });

    // Prompt user to subscribe to notifications
    OneSignal.showSlidedownPrompt();

    // Send user subscription ID to backend
    OneSignal.on('subscriptionChange', async function(isSubscribed) {
      if (isSubscribed) {
        const subscriptionId = await OneSignal.getUserId();
        if (subscriptionId) {
          console.log('OneSignal Subscription ID:', subscriptionId);
          try {
            const response = await fetch('http://localhost:5000/api/save-onesignal-id', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ subscription_id: subscriptionId }),
            });
            const result = await response.json();
            console.log('Saved OneSignal ID:', result);
          } catch (error) {
            console.error('Error saving OneSignal ID:', error);
          }
        }
      }
    });
  });
};