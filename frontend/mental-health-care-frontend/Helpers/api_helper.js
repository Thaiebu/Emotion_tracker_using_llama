import Cookie from "js-cookie";
import axios from "axios";
import { NODE_APP_URL } from "../config/app_config";

// default
axios.defaults.baseURL = NODE_APP_URL;

// content type
axios.defaults.headers.common["Content-Type"] = "application/json";
axios.defaults.withCredentials = true;

const token = Cookie.get("token");
axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

// intercepting to capture errors
axios.interceptors.response.use(
  function (response) {
    return response.data ? response.data : response;
  },
  function (error) {
    if (
      error.response &&
      error.response.status === 401 &&
      window.location.pathname !== "/login"
    ) {
      // Redirect to /login
      Cookie.remove("token");
      window.location.href = "/login";
    }
    return Promise.reject(
      error.response.data ? error.response.data : error.response
    );
  }
);

export class APIClient {
  /**
   * Fetches data from given url
   */
  get = async (url, params) => {
    return await axios.get(url, params);
  };

  /**
   * post given data to url
   */
  create = async (url, data = {}, headers = {}) => {
    return await axios.post(url, data, { headers: headers });
  };

  /**
   * Updates data
   */
  update = async (url, data = {}, headers = {}) => {
    return await axios.put(url, data, { headers: headers });
  };

  /**
   * Delete
   */
  delete = async (url, config = {}) => {
    return await axios.delete(url, { ...config });
  };
}

export const getLoggedinUser = () => {
  let token = Cookie.get("token");
  if (!token) {
    return false;
  } else {
    return token;
  }
};
