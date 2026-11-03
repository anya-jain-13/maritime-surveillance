
/*
 MARIS — CINEMATIC OCEAN HERO
 Procedural top-down cargo ship animation
 No video, image, or external libraries required.
*/

(function () {
  "use strict";

  const CANVAS_ID = "maris-ocean";
  const TWO_PI = Math.PI * 2;

  let canvas = null;
  let ctx = null;
  let hero = null;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let frameId = null;
  let startTime = 0;

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  );

  const random = (min, max) =>
    min + Math.random() * (max - min);

  const waveLines = Array.from(
    { length: 240 },
    () => ({
      x: Math.random(),
      y: Math.random(),
      length: random(10, 90),
      opacity: random(0.025, 0.13),
      speed: random(0.3, 1.2),
      phase: random(0, TWO_PI)
    })
  );

  const foam = Array.from(
    { length: 650 },
    () => ({
      progress: Math.random(),
      side: Math.random() > 0.5 ? 1 : -1,
      spread: random(0.15, 1.15),
      size: random(0.5, 3.5),
      phase: random(0, TWO_PI),
      opacity: random(0.2, 0.9)
    })
  );

  function resize() {
    if (!hero || !canvas || !ctx) return;

    const bounds = hero.getBoundingClientRect();

    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);

    pixelRatio = Math.min(
      window.devicePixelRatio || 1,
      2
    );

    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);

    ctx.setTransform(
      pixelRatio,
      0,
      0,
      pixelRatio,
      0,
      0
    );
  }

  function drawOcean(time) {
    const gradient = ctx.createLinearGradient(
      0,
      0,
      width,
      height
    );

    gradient.addColorStop(0, "#020b17");
    gradient.addColorStop(0.38, "#041827");
    gradient.addColorStop(0.68, "#07506b");
    gradient.addColorStop(1, "#032237");

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Subtle moving patches of reflected light
    for (let i = 0; i < 5; i++) {
      const x =
        width * (0.45 + i * 0.13) +
        Math.sin(time * 0.13 + i) * 35;

      const y =
        height * (0.15 + i * 0.15);

      const glow = ctx.createRadialGradient(
        x,
        y,
        0,
        x,
        y,
        width * 0.35
      );

      glow.addColorStop(
        0,
        "rgba(34,166,201,0.055)"
      );

      glow.addColorStop(
        1,
        "rgba(34,166,201,0)"
      );

      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, width, height);
    }

    // Fine animated ocean texture
    for (const wave of waveLines) {
      const x = wave.x * width;

      const y =
        (
          wave.y * height +
          time * wave.speed * 12
        ) % height;

      const shimmer =
        0.5 +
        0.5 * Math.sin(
          time * 1.5 + wave.phase
        );

      ctx.beginPath();

      ctx.moveTo(x, y);

      ctx.bezierCurveTo(
        x + wave.length * 0.3,
        y - 3,
        x + wave.length * 0.7,
        y + 3,
        x + wave.length,
        y
      );

      ctx.strokeStyle =
        `rgba(145,226,240,${
          wave.opacity * shimmer
        })`;

      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function drawWake(time) {
    const wakeLength = 650;

    // Turbulent center trail
    for (let i = 0; i < 115; i++) {
      const p = i / 115;

      const y = 128 + p * wakeLength;

      const x =
        Math.sin(p * 26 - time * 2.5) *
        (3 + p * 12);

      const radiusX = 8 + p * 52;
      const radiusY = 3 + p * 9;

      ctx.beginPath();

      ctx.ellipse(
        x,
        y,
        radiusX,
        radiusY,
        0,
        0,
        TWO_PI
      );

      ctx.fillStyle =
        `rgba(196,240,250,${
          0.24 * (1 - p)
        })`;

      ctx.fill();
    }

    // Two expanding V-shaped wake arms
    for (const side of [-1, 1]) {
      for (let i = 0; i < 130; i++) {
        const p = i / 130;

        const y = 122 + p * wakeLength;

        const x =
          side * (20 + p * 205) +
          Math.sin(
            p * 38 - time * 3.3
          ) * (2 + p * 9);

        ctx.beginPath();

        ctx.ellipse(
          x,
          y,
          2 + p * 5,
          5 + p * 10,
          side * 0.4,
          0,
          TWO_PI
        );

        ctx.fillStyle =
          `rgba(223,251,255,${
            0.48 * (1 - p)
          })`;

        ctx.fill();
      }
    }

    // Animated foam particles
    for (const particle of foam) {
      const p =
        (
          particle.progress +
          time * 0.055
        ) % 1;

      const y = 120 + p * wakeLength;

      const outer = 14 + p * 200;

      const x =
        particle.side *
        outer *
        particle.spread +
        Math.sin(
          p * 25 + particle.phase
        ) * 7;

      const opacity =
        particle.opacity *
        (1 - p) *
        0.55;

      ctx.beginPath();

      ctx.arc(
        x,
        y,
        particle.size * (0.7 + p),
        0,
        TWO_PI
      );

      ctx.fillStyle =
        `rgba(232,252,255,${opacity})`;

      ctx.fill();
    }
  }

  function drawShip(time) {
    ctx.save();

    // Ship shadow
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 35;
    ctx.shadowOffsetY = 16;

    // Dark steel hull
    ctx.beginPath();

    ctx.moveTo(0, -190);
    ctx.quadraticCurveTo(
      23,
      -184,
      43,
      -158
    );

    ctx.lineTo(48, 120);

    ctx.quadraticCurveTo(
      0,
      150,
      -48,
      120
    );

    ctx.lineTo(-43, -158);

    ctx.quadraticCurveTo(
      -23,
      -184,
      0,
      -190
    );

    ctx.closePath();

    ctx.fillStyle = "#111923";
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    ctx.strokeStyle = "#6c8b99";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Deck
    ctx.fillStyle = "#263644";
    ctx.fillRect(-39, -144, 78, 251);

    // Container grid
    const colors = [
      "#a9543a",
      "#c67647",
      "#415c6b",
      "#697b84",
      "#8c9696",
      "#29465b",
      "#bd7756",
      "#596e79"
    ];

    const containerWidth = 23;
    const containerHeight = 17;

    for (let row = 0; row < 12; row++) {
      for (let col = 0; col < 3; col++) {
        const x = -36 + col * 24;
        const y = -141 + row * 18.5;

        const index =
          (row * 7 + col * 11) %
          colors.length;

        ctx.fillStyle = colors[index];

        ctx.fillRect(
          x,
          y,
          containerWidth,
          containerHeight
        );

        ctx.strokeStyle =
          "rgba(0,0,0,0.45)";

        ctx.lineWidth = 1;

        ctx.strokeRect(
          x,
          y,
          containerWidth,
          containerHeight
        );

        // Container ridges
        ctx.strokeStyle =
          "rgba(255,255,255,0.13)";

        ctx.beginPath();

        for (let k = 5; k < 22; k += 5) {
          ctx.moveTo(x + k, y + 2);
          ctx.lineTo(
            x + k,
            y + containerHeight - 2
          );
        }

        ctx.stroke();
      }
    }

    // Rear superstructure
    ctx.fillStyle = "#d0d8d8";
    ctx.fillRect(-37, 85, 74, 34);

    ctx.fillStyle = "#173346";
    ctx.fillRect(-33, 89, 66, 10);

    ctx.fillStyle = "#9fb5bd";
    ctx.fillRect(-25, 104, 50, 9);

    // Bow deck
    ctx.fillStyle = "#3a4c57";

    ctx.beginPath();
    ctx.moveTo(0, -180);
    ctx.lineTo(29, -151);
    ctx.lineTo(-29, -151);
    ctx.closePath();
    ctx.fill();

    // Navigation lights
    const glow =
      0.65 +
      0.35 * Math.sin(time * 2);

    ctx.fillStyle =
      `rgba(255,94,94,${glow})`;

    ctx.beginPath();
    ctx.arc(-46, -130, 3, 0, TWO_PI);
    ctx.fill();

    ctx.fillStyle =
      `rgba(78,255,176,${glow})`;

    ctx.beginPath();
    ctx.arc(46, -130, 3, 0, TWO_PI);
    ctx.fill();

    ctx.restore();
  }

  function drawCinematicOverlay() {
    // Darken the left side for readable text
    const left = ctx.createLinearGradient(
      0,
      0,
      width,
      0
    );

    left.addColorStop(
      0,
      "rgba(1,8,18,0.92)"
    );

    left.addColorStop(
      0.38,
      "rgba(1,8,18,0.66)"
    );

    left.addColorStop(
      0.65,
      "rgba(1,8,18,0.12)"
    );

    left.addColorStop(
      1,
      "rgba(1,8,18,0.05)"
    );

    ctx.fillStyle = left;
    ctx.fillRect(0, 0, width, height);

    // Bottom cinematic fade
    const bottom = ctx.createLinearGradient(
      0,
      height * 0.55,
      0,
      height
    );

    bottom.addColorStop(
      0,
      "rgba(1,8,18,0)"
    );

    bottom.addColorStop(
      1,
      "rgba(1,8,18,0.65)"
    );

    ctx.fillStyle = bottom;
    ctx.fillRect(0, 0, width, height);
  }

  function render(timestamp) {
    if (!ctx || !hero) return;

    if (!startTime) {
      startTime = timestamp;
    }

    const time =
      (timestamp - startTime) / 1000;

    ctx.clearRect(
      0,
      0,
      width,
      height
    );

    drawOcean(time);

    // Position ship on right side of homepage
    const scale = Math.max(
      0.55,
      Math.min(
        width / 1550,
        height / 800
      ) * 1.5
    );

    const shipX =
      width * 0.77 +
      Math.sin(time * 0.17) * 9;

    const shipY =
      height * 0.44 -
      Math.sin(time * 0.12) * 22;

    ctx.save();

    ctx.translate(shipX, shipY);

    // Small natural heading variation
    ctx.rotate(
      -0.16 +
      Math.sin(time * 0.12) * 0.008
    );

    ctx.scale(scale, scale);

    drawWake(time);
    drawShip(time);

    ctx.restore();

    drawCinematicOverlay();

    if (!reduceMotion.matches) {
      frameId = requestAnimationFrame(render);
    }
  }

  function initialize() {
    hero = document.querySelector(".hero");

    if (!hero) {
      return false;
    }

    if (document.getElementById(CANVAS_ID)) {
      return true;
    }

    canvas = document.createElement("canvas");
    canvas.id = CANVAS_ID;

    canvas.setAttribute(
      "aria-hidden",
      "true"
    );

    hero.prepend(canvas);

    ctx = canvas.getContext("2d");

    if (!ctx) {
      return false;
    }

    resize();

    window.addEventListener(
      "resize",
      resize
    );

    // First frame appears immediately
    render(0);

    if (reduceMotion.matches) {
      cancelAnimationFrame(frameId);
    }

    console.log(
      "MARIS cinematic ocean initialized"
    );

    return true;
  }

  // IMPORTANT:
  // Dash renders the page asynchronously.
  // Wait for the hero element to exist.

  let attempts = 0;

  function waitForDash() {
    if (initialize()) {
      return;
    }

    attempts++;

    if (attempts < 150) {
      setTimeout(waitForDash, 100);
    } else {
      console.error(
        "MARIS ocean: .hero not found"
      );
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener(
      "DOMContentLoaded",
      waitForDash
    );
  } else {
    waitForDash();
  }
})();

