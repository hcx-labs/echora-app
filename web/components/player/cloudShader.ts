/** Separated cloud banks with fine erosion, volumetric extinction, and directional lighting. */
export const cloudFragmentShader = `
precision highp float;
varying vec2 vUv;
uniform vec2 resolution;
uniform vec3 tint, secondaryTint;
uniform float travel, bass, mid, treble, glow, strike, seed;
// The song's journey: forward flight and the sky glowing behind the banks.
uniform float flight, skyLevel;
uniform vec3 skyTint;
float hash(vec3 p) {
  p = fract(p * .3183099 + vec3(.11,.27,.43));
  p *= 17.0;
  return fract(p.x*p.y*p.z*(p.x+p.y+p.z));
}
float noise(vec3 p) {
  vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),
    mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
    mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),
    mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p) {
  float n=0.0, a=.53;
  for(int i=0;i<4;i++) { n+=a*noise(p); p=p*2.03+vec3(11.7,3.1,7.9); a*=.48; }
  return n;
}
vec3 cloudSpace(vec3 p) {
  return p + vec3(travel*.28, -travel*.06, travel*.11 + flight);
}
float cloudBody(vec3 p) {
  // Broad lobes establish distinct masses before smaller folds erode the edges.
  vec3 q=cloudSpace(p);
  float coverage=noise(q*.52+vec3(8.2,1.4,5.7));
  float lobes=noise(q*1.3+vec3(1.7,6.2,3.4));
  float body=coverage*.62+lobes*.38;
  return smoothstep(.42,.62,body)*(1.9+bass*.55);
}
float density(vec3 p) {
  float body=cloudBody(p);
  if(body<.005) return 0.;
  vec3 q=cloudSpace(p);
  float detail=fbm(q*3.1+vec3(0.,travel*.025,0.));
  // No haze floor. The spaces between banks stay dark and reveal their silhouettes.
  float erosion=(1.-detail)*.36;
  return max(0.,body-erosion)*(1.+detail*.35);
}
float boltX(float y, float boltSeed) {
  float cell=y*23.0;
  float jag=mix(hash(vec3(floor(cell),boltSeed,1)),hash(vec3(floor(cell)+1.0,boltSeed,1)),fract(cell));
  return (hash(vec3(boltSeed,2,1))-.5)*1.7 + (jag-.5)*.13 + sin(y*6.0+boltSeed)*.09;
}
float lightning(vec2 screenUv, vec2 aspectUv, float boltSeed) {
  float y=screenUv.y;
  float x=boltX(y,boltSeed);
  float line=abs(aspectUv.x-x);
  float branchY=clamp((.59-y)/.32,0.0,1.0);
  float branch=abs(aspectUv.x-(boltX(.59,boltSeed)+branchY*.27+sin(y*65.0+boltSeed)*.012));
  float branchMask=smoothstep(.24,.30,y)*(1.0-smoothstep(.56,.60,y));
  float bolt=exp(-line*620.0)+exp(-line*55.0)*.22;
  bolt+=(exp(-branch*580.0)*.6+exp(-branch*65.0)*.12)*branchMask;
  return bolt*smoothstep(.10,.24,y)*(1.0-smoothstep(.82,.97,y));
}
void main() {
  vec2 uv=(vUv-.5)*vec2(resolution.x/resolution.y,1.0);
  vec3 ray=normalize(vec3(uv*1.05,1.25));
  vec3 color=vec3(0.0);
  float trans=1.0;
  // Midpoint samples avoid visible screen-space grain in the enlarged backdrop.
  vec3 pigment=clamp(tint,0.0,1.0);
  pigment/=max(.25,max(pigment.r,max(pigment.g,pigment.b)));
  vec3 secondary=clamp(secondaryTint,0.,1.);
  secondary/=max(.25,max(secondary.r,max(secondary.g,secondary.b)));
  vec3 cloudLight=mix(vec3(.78,.81,.84),pigment,.6);
  vec3 fillLight=mix(vec3(.34,.39,.46),secondary,.45);
  vec3 lightPos=vec3((hash(vec3(seed,2,1))-.5)*4.0,.6,2.1);
  // Offset depth samples per pixel so the ray-march planes cannot align into
  // visible vertical or horizontal bands across the cloud field.
  float rayJitter=hash(vec3(gl_FragCoord.xy,0.0))-.5;
  for(int i=0;i<72;i++) {
    float t=(float(i)+.5+rayJitter)*.07;
    vec3 p=vec3(uv*.9,-.6)+ray*t;
    // Banks thin out right in front of the camera, so the flight passes through
    // them instead of meeting a lit wall that fills the screen.
    float d=density(p)*smoothstep(.15,1.1,t);
    if(d<.005) continue;
    vec3 sun=normalize(vec3(-.65,.75,-.45));
    float nearProbe=density(p+sun*.11);
    float farProbe=cloudBody(p+sun*.55);
    float shade=exp(-(nearProbe*.65+farProbe*.9)*1.7);
    float rim=clamp((d-nearProbe)*5.5,0.,1.);
    float powder=1.-exp(-d*2.4);
    // Neutral charcoal stays neutral. Album pigment belongs to lit folds,
    // not a multiplication across every shadow and highlight.
    vec3 smoke=mix(vec3(.008,.010,.015),vec3(.19,.205,.225),shade);
    smoke += fillLight*.013*powder;
    // A faint pigment wash lets artwork hues reach the cloud body without
    // tinting the charcoal shadows or turning the whole screen into a filter.
    smoke += pigment*(shade*.02+rim*.014);
    smoke += cloudLight*rim*(.22+treble*.16)*powder;
    smoke += cloudLight*pow(shade,2.0)*(.025+mid*.11);
    // Light from the sky behind catches the thin edges of each bank.
    smoke += skyTint*skyLevel*(1.-powder)*.35;
    smoke *= .95+bass*.15;
    float distanceToLight=length(p-lightPos);
    float scattered=exp(-distanceToLight*distanceToLight*.65);
    smoke += mix(cloudLight,vec3(.78,.83,.88),.16)*glow*scattered*.44;
    smoke += vec3(.70,.77,.84)*strike*scattered*.45;
    float alpha=1.0-exp(-d*.07*2.4);
    color+=trans*alpha*smoke;
    trans*=1.0-alpha;
    if(trans<.015) break;
  }
  // The sky behind the banks: dark at rest, lit by the journey's colour as the song opens up.
  vec3 sky=skyTint*skyLevel*(.35+.65*smoothstep(-.2,.9,vUv.y))*(1.+.5*exp(-length((vUv-vec2(.5,.62))*vec2(1.6,3.))*2.));
  color+=trans*(vec3(.003,.005,.009)+sky);
  // A normal transient gets one forked bolt. Stronger hits add distant,
  // dimmer bolts so a cluster reads as a storm instead of a white flash.
  float bolt=lightning(vUv,uv,seed);
  bolt+=lightning(vUv,uv,seed+19.37)*.58*smoothstep(.50,.68,strike);
  bolt+=lightning(vUv,uv,seed+43.61)*.34*smoothstep(.66,.80,strike);
  color+=vec3(.46,.53,.64)*bolt*strike;
  color*=1.0-.14*dot(vUv-.5,vUv-.5);
  // Soft highlight rolloff preserves hue when lighting responses overlap.
  color=color/(vec3(1.0)+color*.65);
  gl_FragColor=vec4(color,1.0);
}
`;
