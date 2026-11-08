// tsncd's own webpack configuration, applied to tsncd installed under node_modules/.
// Its ts-loader rule excludes every path under node_modules/, which is where the
// sources now are, so that exclusion is removed and nothing else changes: the entry,
// the output under node_modules/tsncd/dist, the plugins and the dev server are tsncd's.
import tsncd from "tsncd/webpack.config.js";

const rules = tsncd.module.rules.map((rule) => {
  const { exclude, ...rest } = rule;
  return rest;
});

export default { ...tsncd, module: { ...tsncd.module, rules } };
