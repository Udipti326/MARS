import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:5001",
});

const USER_EMAIL_KEY = "mars_user_email";
const USER_NAME_KEY = "mars_user_name";

export function getUserIdentity() {
  let email = localStorage.getItem(USER_EMAIL_KEY);
  let displayName = localStorage.getItem(USER_NAME_KEY);

  if (!email) {
    email = "guest@mars.local";
    localStorage.setItem(USER_EMAIL_KEY, email);
  }

  if (!displayName) {
    displayName = "Local User";
    localStorage.setItem(USER_NAME_KEY, displayName);
  }

  return { email, display_name: displayName };
}

function authHeaders() {
  const { email, display_name } = getUserIdentity();
  return {
    "X-User-Email": email,
    "X-User-Name": display_name,
  };
}

export async function runResearch(query, expeditionId = null) {
  const response = await api.post(
    "/research",
    {
      query,
      expedition_id: expeditionId,
      ...getUserIdentity(),
    },
    { headers: authHeaders() }
  );
  return response.data;
}

export async function saveExpedition(expedition, expeditionId = null) {
  const response = await api.post(
    "/expeditions/save",
    {
      expedition,
      expedition_id: expeditionId,
      ...getUserIdentity(),
    },
    { headers: authHeaders() }
  );
  return response.data;
}

export async function getExpeditions() {
  const { email } = getUserIdentity();
  const response = await api.get("/expeditions", {
    params: { user_email: email },
    headers: authHeaders(),
  });
  return response.data;
}

export async function getExpedition(expeditionId) {
  const response = await api.get(`/expeditions/${expeditionId}`, {
    headers: authHeaders(),
  });
  return response.data;
}

export async function deleteExpedition(expeditionId) {
  const response = await api.delete(`/expeditions/${expeditionId}`, {
    headers: authHeaders(),
  });
  return response.data;
}

export async function getChatMemory(expeditionId) {
  const response = await api.get(`/expeditions/${expeditionId}/chat`, {
    headers: authHeaders(),
  });
  return response.data;
}

export async function sendChatMessage(expeditionId, message) {
  const response = await api.post(
    `/expeditions/${expeditionId}/chat`,
    { message },
    { headers: authHeaders() }
  );
  return response.data;
}

export async function getCFG(expeditionId) {
  const response = await api.get(`/cfg/${expeditionId}`, {
    headers: authHeaders(),
  });
  return response.data;
}

export async function rebuildCFG(expeditionId) {
  const response = await api.post(`/cfg/${expeditionId}/rebuild`, {}, {
    headers: authHeaders(),
  });
  return response.data;
}