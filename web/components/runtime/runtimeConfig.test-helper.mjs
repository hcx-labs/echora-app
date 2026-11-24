import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
const defaults = readFileSync(new URL('./runtimeConfig.defaults.json', import.meta.url), 'utf8');
const source = stripTypeScriptTypes(readFileSync(new URL('./runtimeConfig.ts', import.meta.url), 'utf8'))
  .replace('import defaults from "./runtimeConfig.defaults.json";', `const defaults = ${defaults};`);
export const runtimeConfigModule = `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
