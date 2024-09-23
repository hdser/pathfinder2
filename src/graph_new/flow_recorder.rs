use crate::types::{Address, U256};
use crate::graph_new::graph::FlowGraph;
use std::collections::{HashMap, HashSet};
use gif::{Frame, Encoder, Repeat};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{
    draw_filled_circle_mut,
    draw_line_segment_mut,
    draw_polygon_mut,
};
use imageproc::point::Point;
use rusttype::{Font, Scale};

pub struct FlowRecorder {
    steps: Vec<FlowStep>,
    all_nodes: HashSet<Address>,
}

struct FlowStep {
    current_flow: U256,
    path: Vec<Address>,
    path_flow: U256,
    graph_state: FlowGraph,
}

impl FlowRecorder {
    pub fn new() -> Self {
        FlowRecorder {
            steps: Vec::new(),
            all_nodes: HashSet::new(),
        }
    }

    pub fn record_step(&mut self, current_flow: U256, path: Vec<Address>, path_flow: U256, graph_state: FlowGraph) {
        self.all_nodes.extend(path.iter().cloned());
        self.steps.push(FlowStep {
            current_flow,
            path,
            path_flow,
            graph_state,
        });
    }

    pub fn generate_visualization(&self, output_path: &str, source: Address, sink: Address) -> Result<(), Box<dyn std::error::Error>> {
        let width: u32 = 1600;
        let height: u32 = 900;
        let mut encoder = Encoder::new(std::fs::File::create(output_path)?, width as u16, height as u16, &[])?;
        encoder.set_repeat(Repeat::Infinite)?;

        let node_positions = self.calculate_node_positions(source, sink, width, height);
        let mut previous_paths: Vec<Vec<Address>> = Vec::new();

        for (i, step) in self.steps.iter().enumerate() {
            let mut image = RgbaImage::new(width, height);

            // Clear the image
            for pixel in image.pixels_mut() {
                *pixel = Rgba([255, 255, 255, 255]);
            }

            // Draw previous paths in gray
            for path in &previous_paths {
                self.draw_path(&mut image, path, &Rgba([200, 200, 200, 255]), &node_positions);
            }

            // Draw current path in red
            self.draw_path(&mut image, &step.path, &Rgba([255, 0, 0, 255]), &node_positions);

            // Draw source and sink nodes
            self.draw_node(&mut image, &source, &node_positions, Rgba([0, 255, 0, 255]));
            self.draw_node(&mut image, &sink, &node_positions, Rgba([0, 0, 255, 255]));

            // Add text information
            self.draw_text(&mut image, 10, 10, &format!("Step: {}", i + 1));
            self.draw_text(&mut image, 10, 30, &format!("Current Total Flow: {} tokens", step.current_flow));
            self.draw_text(&mut image, 10, 50, &format!("Path Flow: {} tokens", step.path_flow));
            self.draw_text(&mut image, 10, 70, "Green: Source, Blue: Sink, Red: Active path");
            self.draw_text(&mut image, 10, 90, "Gray: Previous paths");

            // Convert the image to a GIF frame and write it
            let mut frame = Frame::from_rgba(width as u16, height as u16, &mut image.into_raw());
            frame.delay = 200; // 2 seconds delay
            encoder.write_frame(&frame)?;

            // Add current path to previous paths for next iteration
            previous_paths.push(step.path.clone());
        }

        Ok(())
    }

    fn calculate_node_positions(
        &self,
        source: Address,
        sink: Address,
        width: u32,
        height: u32,
    ) -> HashMap<Address, (u32, u32)> {
        let mut node_positions = HashMap::new();
        let margin = 100u32;
    
        // Build node levels mapping
        let mut node_levels = HashMap::new();
    
        // Process all paths to assign levels to nodes
        for step in &self.steps {
            for (index, &node) in step.path.iter().enumerate() {
                // Assign the earliest level
                node_levels
                    .entry(node)
                    .and_modify(|e| if index < *e { *e = index })
                    .or_insert(index);
            }
        }
    
        // Adjust levels so that source is at level 0
        node_levels.insert(source, 0);
    
        // Adjust levels so that sink is at the highest level + 1
        let max_level = node_levels.values().max().cloned().unwrap_or(0);
        let sink_level = max_level + 1;
        node_levels.insert(sink, sink_level);
    
        // Update max_level after inserting sink
        let max_level = sink_level;
    
        // Build levels to nodes mapping
        let mut levels_to_nodes: HashMap<usize, Vec<Address>> = HashMap::new();
        for (&node, &level) in &node_levels {
            levels_to_nodes.entry(level).or_default().push(node);
        }
    
        // Assign positions based on levels
        let level_width = (width - 2 * margin) / (max_level as u32);
    
        for level in 0..=max_level {
            let x = margin + (level as u32 * level_width);
            if let Some(nodes_at_level) = levels_to_nodes.get(&level) {
                let num_nodes = nodes_at_level.len() as u32;
                for (i, &node) in nodes_at_level.iter().enumerate() {
                    // Spread nodes vertically
                    let y = margin + ((height - 2 * margin) / (num_nodes + 1)) * (i as u32 + 1);
                    node_positions.insert(node, (x, y));
                }
            }
        }
    
        node_positions
    }

    fn draw_node(
        &self,
        image: &mut RgbaImage,
        node: &Address,
        node_positions: &HashMap<Address, (u32, u32)>,
        color: Rgba<u8>,
    ) {
        if let Some(&(x, y)) = node_positions.get(node) {
            // Cast to i32 where necessary
            draw_filled_circle_mut(image, (x as i32, y as i32), 20, color);
    
            // Safely calculate text positions
            let text_x = x.saturating_sub(15);
            let text_y = y.saturating_sub(7);
    
            self.draw_text(image, text_x, text_y, &node.short());
        }
    }

    fn draw_path(
        &self,
        image: &mut RgbaImage,
        path: &[Address],
        color: &Rgba<u8>,
        node_positions: &HashMap<Address, (u32, u32)>,
    ) {
        for window in path.windows(2) {
            let from = window[0];
            let to = window[1];
            if let (Some(&(x1, y1)), Some(&(x2, y2))) = (node_positions.get(&from), node_positions.get(&to)) {
                self.draw_arrow(image, (x1, y1), (x2, y2), *color);
    
                // Draw nodes of the path
                self.draw_node(image, &from, node_positions, *color);
                self.draw_node(image, &to, node_positions, *color);
            }
        }
    }

    fn draw_arrow(&self, image: &mut RgbaImage, from: (u32, u32), to: (u32, u32), color: Rgba<u8>) {
        let dx = to.0 as f32 - from.0 as f32;
        let dy = to.1 as f32 - from.1 as f32;
        let length = (dx * dx + dy * dy).sqrt();
        let unit_x = dx / length;
        let unit_y = dy / length;
    
        // Calculate midpoint
        let mid_x = (from.0 as f32 + to.0 as f32) / 2.0;
        let mid_y = (from.1 as f32 + to.1 as f32) / 2.0;
    
        let arrow_size: f32 = 15.0;
        let arrow_angle: f32 = 0.5;
    
        // Draw the main line of the edge
        draw_line_segment_mut(
            image,
            (from.0 as f32, from.1 as f32),
            (to.0 as f32, to.1 as f32),
            color,
        );
    
        // Calculate arrowhead position at the midpoint
        let tip_x = mid_x;
        let tip_y = mid_y;
    
        let left_x = tip_x - arrow_size * (unit_x * arrow_angle.cos() + unit_y * arrow_angle.sin());
        let left_y = tip_y - arrow_size * (-unit_x * arrow_angle.sin() + unit_y * arrow_angle.cos());
        let right_x = tip_x - arrow_size * (unit_x * arrow_angle.cos() - unit_y * arrow_angle.sin());
        let right_y = tip_y - arrow_size * (unit_x * arrow_angle.sin() + unit_y * arrow_angle.cos());
    
        let arrow_points = vec![
            Point::new(tip_x as i32, tip_y as i32),
            Point::new(left_x as i32, left_y as i32),
            Point::new(right_x as i32, right_y as i32),
        ];
    
        // Draw the arrowhead
        draw_polygon_mut(image, &arrow_points, color);
    }

    fn draw_text(&self, image: &mut RgbaImage, x: u32, y: u32, text: &str) {
        // Load the font data
        let font_data = include_bytes!("../assets/Roboto/Roboto-Black.ttf") as &[u8];
        let font = Font::try_from_bytes(font_data).unwrap();
        let scale = Scale::uniform(24.0); // Adjust font size as needed
        let color = Rgba([0, 0, 0, 255]);
    
        // Pass x and y as u32
        imageproc::drawing::draw_text_mut(image, color, x, y, scale, &font, text);
    }
}
