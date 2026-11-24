export type ShellUser = {
  username: string;
  display_name: string;
  onboarding_complete: boolean;
  navidrome_connection_id?: string | null;
};
let cachedUser: ShellUser | null = null;
let generation = 0;
export const SESSION_EXPIRED_EVENT = "echora:session-expired";
export const getCachedUser = () => cachedUser;
export const sessionGeneration = () => generation;
export function cacheUser(user: ShellUser) {
  cachedUser = user;
}
export function invalidateSessionUser() {
  cachedUser = null;
  generation++;
  window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
}
